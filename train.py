import argparse
import sys
import random
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np
import os

from model.spatial import spatial_attention
from model.temporal import temporal_attention
from model.gallat import gallat
from model.transferring import transferring_attention
from utils.utils import get_graph, load_geo_neighbors, load_OD_matrix, name_with_datetime, get_graph_from_distance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset to choose', default='bj')
    parser.add_argument('--batch_size', type=int, default=2020, help='The batch size (defaults to 20)')
    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs')
    parser.add_argument('--loss-weight', type=float, default=0.8, help='The value of loss_d')
    parser.add_argument('--gpu', type=int, help='gpu to use', default=2)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    device = 'cuda:' + str(args.gpu)
    batch_size = args.batch_size
    random_seed = args.random_seed
    learning_rate = args.lr
    epochs = args.epochs

    random.seed(random_seed)  # 设置随机数种子
    torch.cuda.manual_seed_all(random_seed)

    geo_thr = 3
    time_slot = 7

    # data path
    print('Loading data... ', end='')
    if args.dataset == 'ny':
        m_size = 63
        train_day = 139
        vali_day = 21
        test_day = 21
        batch_no = 48
        start_hour = 0
        end_hour = 48
        data_path = '/home/zengjy/data/Manhattan/'
        data = np.load(data_path + 'Manhattan.npy')
        graph = get_graph_from_distance(data_path + 'distance.npy', 4)
    else:
        m_size = 268
        train_day = 42
        vali_day = 7
        test_day = 7
        batch_no = 32
        start_hour = 12
        end_hour = 44
        data_path = '/home/zengjy/data/Beijing/'
        data = np.load(data_path + 'all.npy')
        graph = get_graph(data_path + 'graph.npy')

    geo_neighbors = load_geo_neighbors(graph, m_size, geo_thr)

    feature_dim = m_size * 2 + 2
    embed_dim = 16

    spatial = spatial_attention(m_size, feature_dim, embed_dim, device)
    temporal = temporal_attention(feature_dim, 4 * embed_dim, device)
    transferring = transferring_attention(m_size, 4 * embed_dim, 8 * embed_dim, device)

    gallat = gallat(device, epochs, random_seed, args.lr, batch_size, m_size, feature_dim, embed_dim, batch_no,
                    time_slot, graph, spatial, temporal, transferring).to(device=device)

    gallat.fit(args.dataset, data, epochs, train_day, vali_day, test_day)

    print("Finished.")
