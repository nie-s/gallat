import argparse
import sys
import random
from collections import defaultdict

import torch
import numpy as np

from utils.utils import getGraph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset to choose', default='ny')
    parser.add_argument('--batch-size', type=int, default=20, help='The batch size (defaults to 20)')
    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs')
    parser.add_argument('--loss-weight', type=float, default=0.8, help='The value of loss_d')
    parser.add_argument('--gpu', type=int, help='gpu to use', default=0)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')

    parser.add_argument('--model_no', type=int)
    parser.add_argument('--model_name', type=str, default='test')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    m_size = 268
    train_day = 42
    vali_day = 7
    test_day = 7
    geo_thr = 3

    device = 'cuda:' + str(args.gpu)
    model_name = args.model_name
    model_no = args.model_no
    random_seed = args.random_seed
    learning_rate = args.lr

    random.seed(random_seed)  # 设置随机数种子
    torch.cuda.manual_seed_all(random_seed)

    print('Loading data... ', end='')
    if args.dataset == 'ny':  # TODO data_path
        data_path = 'ny'
    else:
        data_path = 'bj'

    data = np.load(data_path + 'all.npy')
    graph = getGraph(data_path + 'graph.npy')

    geo_neighbors = defaultdict(dict)

    for grid_no in range(0, m_size):
        gn_grid = {}
        for j in range(0, m_size):
            if grid_no != j and graph[grid_no, j] < geo_thr:
                gn_grid[j] = graph[grid_no, j]
        geo_neighbors[grid_no] = gn_grid

