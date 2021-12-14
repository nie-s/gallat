from collections import defaultdict

import torch

from utils.utils import load_forward_neighbors, load_backward_neighbors, load_geo_neighbors, get_graph, pre_weight, \
    pre_weight_geo
import numpy as np


def load_empty(m_size):
    graph = np.zeros([m_size, m_size])
    forward_neighbors = defaultdict(dict)
    for grid_no in range(0, m_size):
        gn_grid = {}
        forward_neighbors[grid_no] = gn_grid

    return graph, forward_neighbors


def get_ne():
    neighbor_list = [[0] * 100 for i in range(100)]
    mask_list = [[0] * 100 for i in range(100)]
    data_path = "/home/hlz/maxj/data/Beijing/"
    data = np.load(data_path + 'all.npy')
    graph = get_graph(data_path + 'graph.npy')

    train_day = 42
    vali_day = 7
    test_day = 7
    batch_no = 32
    start_hour = 12
    end_hour = 44
    m_size = 268
    t = torch.eye(m_size)

    for day in range(train_day + vali_day + test_day):
        for hour in range(0, 49):
            # if start_hour <= hour and hour <= end_hour:
            feat_out = data[day, start_hour]
            forward_adj, forward_neighbors = load_forward_neighbors(feat_out, m_size=m_size)  # check
            backward_adj, backward_neighbors = load_backward_neighbors(feat_out, m_size=m_size)  # check
            geo_neighbors = load_geo_neighbors(graph, m_size=m_size, geo_thr=3)  # check\
            neighbor_list[day][hour] = [forward_adj, forward_neighbors, backward_adj, backward_neighbors,
                                        geo_neighbors]

            mask_forward = torch.mm(pre_weight(forward_neighbors, m_size), t).detach()
            mask_backward = torch.mm(pre_weight(backward_neighbors, m_size), t).detach()
            mask_geo = torch.mm(pre_weight_geo(geo_neighbors, m_size), t).detach()

            mask_list[day][hour] = [mask_forward, mask_backward, mask_geo]
    np.save("bj1.npy", neighbor_list)
    np.save("bj2.npy", mask_list)


get_ne()
