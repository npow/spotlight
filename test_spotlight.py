import argparse
import csv
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


def get_ws_mapping(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser()
parser.register("type", "bool", str2bool)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--embedding_dim", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
parser.add_argument("--l2", type=float, default=1e-5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sparse", type=str2bool, default=False)
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--input_file", type=str, default="filtered_ratings.csv")
parser.add_argument("--ws_file", type=str, default="wine_style_mapping.pkl")
parser.add_argument("--loss", type=str, default="regression")
parser.add_argument("--reserved_user_ids", type=int, default=1000)


def main(
    input_file,
    ws_file,
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
    ws_mapping = get_ws_mapping(ws_file)
    user_ids, wine_ids, ratings = zip(*L)
    # with open('ratings.pkl', 'rb') as f:
    #    user_ids, wine_ids, ratings = pickle.load(f)

    uniq_user_ids = sorted(set(user_ids))
    uniq_wine_ids = sorted(set(wine_ids))
    user_id_mapping = {user_id: i for i, user_id in enumerate(uniq_user_ids)}
    wine_id_mapping = {wine_id: i for i, wine_id in enumerate(uniq_wine_ids)}
    user_idxs = np.array([user_id_mapping[x] for x in user_ids])
    wine_idxs = np.array([wine_id_mapping[x] for x in wine_ids])
    ratings = np.array([(r-1.)/4. for r in ratings])

    ws_ids = [ws_mapping[wine_id] for wine_id in uniq_wine_ids]
    uniq_ws_ids = sorted(set(ws_ids))
    ws_id_mapping = {ws_id: i for i, ws_id in enumerate(ws_ids)}
    ws_idxs = np.array([ws_id_mapping[x] for x in ws_ids]).reshape((-1, 1))

    num_users = len(uniq_user_ids) + reserved_user_ids
    dataset = Interactions(user_ids=user_idxs, item_ids=wine_idxs, ratings=ratings, item_features=ws_idxs, num_users=num_users)
    random_state = np.random.RandomState(seed)
    train, test = random_train_test_split(dataset, random_state=random_state, test_percentage=0.1)

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
        user_id_mapping=user_id_mapping,
        wine_id_mapping=wine_id_mapping,
        ws_id_mapping=ws_id_mapping,
        loss=loss,
    )
    for epoch in range(num_epochs):
        model.fit(train, verbose=True)
        torch.save(model, f"{checkpoint_dir}/model_{epoch:04d}.pt")
        continue
        with torch.no_grad():
            train_rmse = rmse_score(model, train)
            test_rmse = rmse_score(model, test)
            print("         Train RMSE {:.3f}, Test RMSE {:.3f}".format(train_rmse, test_rmse))


if __name__ == "__main__":
    args = parser.parse_args()
    print(args.__dict__)
    main(**args.__dict__)
