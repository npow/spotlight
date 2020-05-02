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
from spotlight.factorization.representations import BilinearNet


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print('start: {}'.format(self.name))
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        print('end: {}, took: {}'.format(self.name, self.interval))


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
    

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.register('type', 'bool', str2bool)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--embedding_dim', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
parser.add_argument('--l2', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--sparse', type=str2bool, default=False)


def main(batch_size, embedding_dim, checkpoint_dir, num_epochs, l2, lr, seed, sparse):
    L = get_ratings('filtered_ratings.csv')
    user_ids, wine_ids, ratings = zip(*L)
    #with open('ratings.pkl', 'rb') as f:
    #    user_ids, wine_ids, ratings = pickle.load(f)

    user_id_mapping = {user_id:i for i, user_id in enumerate(sorted(set(user_ids)))}
    wine_id_mapping = {wine_id:i for i, wine_id in enumerate(sorted(set(wine_ids)))}
    user_idxs = np.array([user_id_mapping[x] for x in user_ids])
    wine_idxs = np.array([wine_id_mapping[x] for x in wine_ids])
    ratings = np.array(ratings)

    dataset = Interactions(user_ids=user_idxs, item_ids=wine_idxs, ratings=ratings)
    random_state = np.random.RandomState(seed)
    train, test = random_train_test_split(dataset, random_state=random_state)

    representation = BilinearNet(dataset.num_users, dataset.num_items, embedding_dim, sparse=sparse)
    representation = NCF(dataset.num_users, dataset.num_items, embedding_dim, layers=[2*embedding_dim, embedding_dim], dropout=0.0)
    if torch.cuda.device_count() > 1:
        representation = nn.DataParallel(representation)

    model = ExplicitFactorizationModel(n_iter=1, l2=l2, learning_rate=lr, embedding_dim=embedding_dim, use_cuda=True, batch_size=batch_size, representation=representation, sparse=sparse, random_state=random_state)
    for epoch in range(num_epochs):
        model.fit(train, verbose=True)
        torch.save(model, f'{checkpoint_dir}/model_{epoch:04d}.pt')
        continue
        with torch.no_grad():
            train_rmse = rmse_score(model, train)
            test_rmse = rmse_score(model, test)
            print('         Train RMSE {:.3f}, Test RMSE {:.3f}'.format(train_rmse, test_rmse))


if __name__ == '__main__':
  args = parser.parse_args()
  print(args.__dict__)
  main(**args.__dict__)
