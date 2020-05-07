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
from keras_preprocessing.sequence import pad_sequences



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
    item_features = mlb.transform(wine_features).tolil().rows
    item_features = pad_sequences(item_features, maxlen=None, dtype='int64', padding='pre', truncating='pre', value=0)
    return item_features


def transform_rating(r):
    if r < 4:
        return 3.5
    if r > 4:
        return 4.5
    return 4.0


parser = argparse.ArgumentParser()
parser.register("type", "bool", str2bool)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--embedding_dim", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
parser.add_argument("--l2", type=float, default=0.02)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sparse", type=str2bool, default=False)
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--input_file", type=str, default="filtered_ratings.csv")
parser.add_argument("--wf_file", type=str, default="wine_feature_mapping.pkl")
parser.add_argument("--loss", type=str, default="regression")
parser.add_argument("--reserved_user_ids", type=int, default=1000)


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
    user_ids, wine_ids, ratings = zip(*L)
    # with open('ratings.pkl', 'rb') as f:
    #    user_ids, wine_ids, ratings = pickle.load(f)

    uniq_user_ids = sorted(set(user_ids))
    uniq_wine_ids = sorted(set(wine_ids))
    user_id_mapping = {user_id: i for i, user_id in enumerate(uniq_user_ids)}
    wine_id_mapping = {wine_id: i for i, wine_id in enumerate(uniq_wine_ids)}
    user_idxs = np.array([user_id_mapping[x] for x in user_ids])
    wine_idxs = np.array([wine_id_mapping[x] for x in wine_ids])
    #ratings = np.array([(r-1.)/4. for r in ratings], dtype=np.float32)
    ratings = np.array([transform_rating(r) for r in ratings], dtype=np.float32)
    mu = ratings.mean()


    wine_features = [wf_mapping[wine_id] for wine_id in uniq_wine_ids]
    uniq_wine_features = set()
    for wfs in wine_features:
        uniq_wine_features |= set(wfs)
    uniq_wine_features = ['<pad>'] + list(sorted(uniq_wine_features))

    mlb = MultiLabelBinarizer(sparse_output=True, classes=uniq_wine_features)
    mlb.fit(wine_features)

    item_features = get_wfs(wf_mapping, uniq_wine_ids, mlb)

    num_users = len(uniq_user_ids) + reserved_user_ids
    dataset = Interactions(user_ids=user_idxs, item_ids=wine_idxs, ratings=ratings, item_features=item_features, num_users=num_users)
    random_state = np.random.RandomState(seed)
    train, test = random_train_test_split(dataset, random_state=random_state, test_percentage=0.1)
    all_item_ids = list(uniq_wine_ids)

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
        mu=mu,
    )

    with open(f"{checkpoint_dir}/mappings.pkl", "wb") as f:
        mappings = {
            "user_id_mapping": user_id_mapping,
            "wine_id_mapping": wine_id_mapping,
            "wf_mapping": wf_mapping,
            "mlb": mlb,
        }
        pickle.dump(mappings, f)

    #train_item_features = get_wfs(wf_mapping, [all_item_ids[x] for x in train.item_ids], mlb)
    test_item_features = get_wfs(wf_mapping, [all_item_ids[x] for x in test.item_ids], mlb)
    for epoch in range(num_epochs):
        model.fit(train, verbose=True)
        print(model._net.mu)
        print(model._net.latent.user_biases.weight.data.max(), model._net.latent.user_biases.weight.data.min())
        print(model._net.latent.item_biases.weight.data.max(), model._net.latent.item_biases.weight.data.min())
        #torch.save(model, f"{checkpoint_dir}/model_{epoch:04d}.pt")
        with torch.no_grad():
            predictions = model.predict(test.user_ids, test.item_ids, item_features=test_item_features)
            print('test rmse: ', np.sqrt((((test.ratings) - (predictions)) ** 2).mean()))

            #predictions = model.predict(train.user_ids, train.item_ids, item_features=train_item_features)
            #print('train rmse: ', np.sqrt((((train.ratings) - (predictions)) ** 2).mean()))




if __name__ == "__main__":
    args = parser.parse_args()
    print(args.__dict__)
    main(**args.__dict__)
