import argparse
import csv
import itertools
import numpy as np
import pickle
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import *
from keras_preprocessing.sequence import pad_sequences
from collections import defaultdict



class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print("start: {}".format(self.name))
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        print("end: {}, took: {}".format(self.name, self.interval))


def get_ratings(fname):
    L = []
    with open(fname) as f:
        reader = csv.reader(f)
        for user_id, wine_id, rating in reader:
            user_id = int(user_id)
            wine_id = int(wine_id)
            rating = float(rating)
            L.append((user_id, wine_id, rating))
    return L


def get_wf_mapping(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_wfs(wf_mapping, wine_ids, mlb):
    wine_features = [wf_mapping[wine_id] for wine_id in wine_ids]
    features = mlb.transform(wine_features).tolil().rows
    features = pad_sequences(features, maxlen=None, dtype='int64', padding='pre', truncating='pre', value=0)
    return features


def get_ufs(uf_mapping, user_ids, mlb):
    user_features = [uf_mapping[user_id] for user_id in user_ids]
    features = mlb.transform(user_features).tolil().rows
    features = pad_sequences(features, maxlen=None, dtype='int64', padding='pre', truncating='pre', value=0)
    return features


def transform_ratings(r):
    r = np.clip(r, 3.0, 4.5)
    return r


def rating_to_label(r):
    if r < 4:
        return 0
    return 1


parser = argparse.ArgumentParser()
parser.register("type", "bool", str2bool)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--embedding_dim", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
parser.add_argument("--l2", type=float, default=0.00)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sparse", type=str2bool, default=False)
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--input_file", type=str, default="filtered_ratings.csv")
parser.add_argument("--wf_file", type=str, default="wine_feature_mapping_no_id.pkl")
parser.add_argument("--loss", type=str, default="bce")
parser.add_argument("--reserved_user_ids", type=int, default=0)


def get_uf_mapping(L, wf_mapping):
    uf_mapping = defaultdict(set)
    for user_id, wine_id, rating in tqdm(L):
        features = set()
        if rating >= 4:
            features |= set([wf for wf in wf_mapping[wine_id] if not wf.startswith('id:')])
        uf_mapping[user_id] |= features
    for k, vs in uf_mapping.items():
        uf_mapping[k] = list(vs)
    return uf_mapping


def get_user_mlb(uf_mapping, uniq_user_ids):
    user_features = [uf_mapping[user_id] for user_id in uniq_user_ids]
    uniq_user_features = set()
    for ufs in user_features:
        uniq_user_features |= set(ufs)
    uniq_user_features = ['<pad>'] + list(sorted(uniq_user_features))

    user_mlb = MultiLabelBinarizer(sparse_output=True, classes=uniq_user_features)
    user_mlb.fit(user_features)
    return user_mlb


def get_wine_mlb(wf_mapping, uniq_wine_ids):
    wine_features = [wf_mapping[wine_id] for wine_id in uniq_wine_ids]
    uniq_wine_features = set()
    for wfs in wine_features:
        uniq_wine_features |= set(wfs)
    uniq_wine_features = ['<pad>'] + list(sorted(uniq_wine_features))

    wine_mlb = MultiLabelBinarizer(sparse_output=True, classes=uniq_wine_features)
    wine_mlb.fit(wine_features)
    return wine_mlb


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main(
    input_file,
    wf_file,
    batch_size,
    embedding_dim,
    checkpoint_dir,
    num_epochs,
    l2,
    lr,
    seed,
    sparse,
    use_cuda,
    loss,
    reserved_user_ids,
):
    L = get_ratings(input_file)
    wf_mapping = get_wf_mapping(wf_file)
    uf_mapping = get_uf_mapping(L, wf_mapping)
    user_ids, wine_ids, ratings = zip(*L)
    # with open('ratings.pkl', 'rb') as f:
    #    user_ids, wine_ids, ratings = pickle.load(f)

    uniq_user_ids = sorted(set(user_ids))
    uniq_wine_ids = sorted(set(wine_ids))
    user_id_mapping = {user_id: i for i, user_id in enumerate(uniq_user_ids)}
    wine_id_mapping = {wine_id: i for i, wine_id in enumerate(uniq_wine_ids)}
    print('num users: ', len(uniq_user_ids), ' num_wines: ', len(uniq_wine_ids))
    user_idxs = np.array([user_id_mapping[x] for x in user_ids])
    wine_idxs = np.array([wine_id_mapping[x] for x in wine_ids])
    ratings = np.array(ratings)
    if loss == 'bce':
        ratings = np.array([rating_to_label(r) for r in ratings], dtype=np.int32)
    else:
        ratings = transform_ratings(ratings)
        scaler = StandardScaler(with_std=False)
        ratings = scaler.fit_transform(ratings.reshape((-1, 1))).reshape((-1,))
        mu = ratings.mean()
        print('min: ', ratings.min(), 'max: ', ratings.max(), 'mean: ', mu, 'scaler: ', scaler.mean_)

    wine_mlb = get_wine_mlb(wf_mapping, uniq_wine_ids)
    user_mlb = get_user_mlb(uf_mapping, uniq_user_ids)

    item_features = get_wfs(wf_mapping, uniq_wine_ids, wine_mlb)
    user_features = get_ufs(uf_mapping, uniq_user_ids, user_mlb)

    num_users = len(uniq_user_ids) + reserved_user_ids
    dataset = Interactions(user_ids=user_idxs, item_ids=wine_idxs, ratings=ratings,
            user_features=user_features, item_features=item_features, num_users=num_users)
    random_state = np.random.RandomState(seed)
    train, test = random_train_test_split(dataset, random_state=random_state, test_percentage=0.1)
    all_item_ids = list(uniq_wine_ids)
    all_user_ids = list(uniq_user_ids)

    if torch.cuda.device_count() > 1:
        representation = nn.DataParallel(representation)

    model = ExplicitFactorizationModel(
        n_iter=1,
        l2=l2,
        learning_rate=lr,
        embedding_dim=embedding_dim,
        use_cuda=use_cuda,
        batch_size=batch_size,
        sparse=sparse,
        random_state=random_state,
        layers=[2*embedding_dim, embedding_dim],
        loss=loss,
        user_mlb=user_mlb,
        wine_mlb=wine_mlb,
    )

    with open(f"{checkpoint_dir}/mappings.pkl", "wb") as f:
        mappings = {
            "user_id_mapping": user_id_mapping,
            "wine_id_mapping": wine_id_mapping,
            "wf_mapping": wf_mapping,
            "wine_mlb": wine_mlb,
        }
        pickle.dump(mappings, f)

    test_item_features = get_wfs(wf_mapping, [all_item_ids[x] for x in test.item_ids], wine_mlb)
    test_user_features = get_ufs(uf_mapping, [all_user_ids[x] for x in test.user_ids], user_mlb)
    for epoch in range(num_epochs):
        model.fit(train, verbose=True)
        print(model._net.mu)
        print(model._net.latent.user_biases.weight.data.max(), model._net.latent.user_biases.weight.data.min())
        print(model._net.latent.item_biases.weight.data.max(), model._net.latent.item_biases.weight.data.min())
        torch.save(model, f"{checkpoint_dir}/model_{epoch:04d}.pt")
        with torch.no_grad():
            predictions = []
            indices = np.arange(len(test.user_ids))
            for batch_indices in tqdm(chunks(indices, batch_size), total=len(indices)//batch_size):
                batch_user_ids = test.user_ids[batch_indices]
                batch_item_ids = test.item_ids[batch_indices]
                preds = model.predict(batch_user_ids, batch_item_ids,
                            user_features=get_ufs(uf_mapping, [all_user_ids[x] for x in batch_user_ids], user_mlb),
                            item_features=get_wfs(wf_mapping, [all_item_ids[x] for x in batch_item_ids], wine_mlb))
                predictions.extend(preds)
            if loss == 'bce':
                labels = [1 if p > 0.5 else 0 for p in predictions]
                print(classification_report(test.ratings.astype(np.int32), labels))
            else:
                print(scaler.inverse_transform(test.ratings[:10]))
                print(scaler.inverse_transform(predictions[:10]))
                print('test rmse: ', np.sqrt(((scaler.inverse_transform(test.ratings) - scaler.inverse_transform(predictions)) ** 2).mean()))



if __name__ == "__main__":
    args = parser.parse_args()
    print(args.__dict__)
    main(**args.__dict__)
