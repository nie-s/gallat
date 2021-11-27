import argparse
import sys
import random
from collections import defaultdict

import torch
import numpy as np

from model.spatial import spatial_attention
from model.temporal import temporal_attention
from utils.utils import get_graph, load_geo_neighbors, load_OD_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset to choose', default='ny')
    parser.add_argument('--batch_size', type=int, default=20, help='The batch size (defaults to 20)')
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
    batch_size = args.batch_size
    model_name = args.model_name
    model_no = args.model_no
    random_seed = args.random_seed
    learning_rate = args.lr

    random.seed(random_seed)  # 设置随机数种子
    torch.cuda.manual_seed_all(random_seed)

    print('Loading data... ', end='')
    if args.dataset == 'ny':
        data_path = '/home/zengjy/data/Manhattan'
    else:
        data_path = '/home/zengjy/data/Beijing/'

    data = np.load(data_path + 'all.npy')
    graph = get_graph(data_path + 'graph.npy')
    geo_neighbors = load_geo_neighbors(graph, m_size, geo_thr)
    batches = random.sample(range(12, 44), batch_size)

    batch_nodes = list(range(m_size))
    feature_dim = m_size * 2 + 2
    embed_dim = 16

    enc1 = spatial_attention(feature_dim, embed_dim, geo_neighbors)

    #########################
    feat_data, feat_out = load_OD_matrix(data, 0, batches)

    feat_out = torch.FloatTensor(feat_out).to(device=device)
    feat_data = torch.FloatTensor(feat_data).to(device=device)
