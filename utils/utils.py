from collections import defaultdict

import numpy as np


def get_graph(graph_path):
    return np.load(graph_path)


def load_geo_neighbors(graph, m_size, geo_thr):
    geo_neighbors = defaultdict(dict)

    for grid_no in range(0, m_size):
        gn_grid = {}
        for j in range(0, m_size):
            if grid_no != j and graph[grid_no, j] < geo_thr:
                gn_grid[j] = graph[grid_no, j]
        geo_neighbors[grid_no] = gn_grid

    return geo_neighbors


def load_forward_neighbors(feat_out, m_size):
    forward_neighbors = defaultdict(dict)
    for grid_no in range(0, m_size):
        gn_grid = {}
        for j in range(0, m_size):
            if grid_no != j and feat_out[grid_no, j] > 0:
                gn_grid[j] = feat_out[grid_no, j]
        forward_neighbors[grid_no] = gn_grid

    return forward_neighbors


def load_backward_neighbors(feat_out, m_size):
    backward_neighbors = defaultdict(dict)
    for grid_no in range(0, m_size):
        gn_grid = {}
        for j in range(0, m_size):
            if grid_no != j and feat_out[j, grid_no] > 0:
                gn_grid[j] = feat_out[grid_no, j]
        backward_neighbors[grid_no] = gn_grid

    return backward_neighbors


def load_OD_matrix(array, day, hours):
    feat_out_list = []
    feat_data_list = []
    for hour in hours:
        feat_out = array[day, hour]
        feat_in = feat_out.T
        feat_o = feat_out.sum(axis=1, keepdims=True)
        feat_i = feat_in.sum(axis=1, keepdims=True)
        feat_data = np.concatenate((feat_out, feat_in, feat_o, feat_i), axis=1)

        feat_out_list.append(feat_out.reshape([1, 268, 268]))
        feat_data_list.append(feat_data.reshape([1, 268, 538]))

    feat_data_all = np.concatenate(feat_data_list, axis=0)
    feat_out_all = np.concatenate(feat_out_list, axis=0)
    return feat_data_all, feat_out_all
