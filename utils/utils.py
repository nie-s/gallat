import math
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from torch.autograd import Variable


def get_graph(graph_path):
    return np.load(graph_path)


def get_graph_from_distance(data_path, k):
    distance = np.load(data_path)
    graph = np.zeros([distance.shape[0], distance.shape[1]])
    median = np.median(distance)
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            if i == j:
                graph[i, j] = 0
            else:
                fen = int(distance[i, j] / (median / k - 1)) + 1
                if fen > k:
                    fen = k
                graph[i, j] = fen
    return graph


def load_geo_neighbors(graph, m_size, geo_thr):
    geo_neighbors = defaultdict(dict)

    for grid_no in range(0, m_size):
        gn_grid = {}
        for j in range(0, m_size):
            if grid_no != j and graph[grid_no, j] < geo_thr:
                gn_grid[j] = 1 / graph[grid_no, j]

        geo_neighbors[grid_no] = gn_grid

    return geo_neighbors


def load_forward_neighbors(feat_out, m_size):
    graph = np.zeros([m_size, m_size])
    forward_neighbors = defaultdict(dict)
    for grid_no in range(0, m_size):
        gn_grid = {}
        for j in range(0, m_size):
            if grid_no != j and feat_out[grid_no, j] > 0:
                graph[grid_no, j] = feat_out[grid_no, j]
                gn_grid[j] = feat_out[grid_no, j]  # todo 这里写成了grad_no

        forward_neighbors[grid_no] = gn_grid  # todo 这里缩进有问题，每一个相当于只找了一个邻居

    return graph, forward_neighbors


def load_backward_neighbors(feat_out, m_size):
    graph = np.zeros([m_size, m_size])
    backward_neighbors = defaultdict(dict)
    for grid_no in range(0, m_size):
        gn_grid = {}
        for j in range(0, m_size):
            if grid_no != j and feat_out[j, grid_no] > 0:
                graph[grid_no, j] = feat_out[j, grid_no]
                gn_grid[j] = feat_out[j, grid_no]
        backward_neighbors[grid_no] = gn_grid

    return graph, backward_neighbors


def load_OD_matrix(array, day, hours):
    feat_out_list = []
    feat_data_list = []
    for hour in hours:
        feat_out = array[day, hour]
        mylen = np.zeros([len(feat_out), len(feat_out)])
        # if (feat_out == mylen).all():
        #     print("+1")
        # else:
        #     print("-1")
        feat_in = feat_out.T
        feat_o = feat_out.sum(axis=1, keepdims=True)
        feat_i = feat_in.sum(axis=1, keepdims=True)
        feat_data = np.concatenate((feat_out, feat_in, feat_o, feat_i), axis=1)

        feat_out_list.append(feat_out.reshape([1, 268, 268]))
        feat_data_list.append(feat_data.reshape([1, 268, 538]))

    feat_data_all = np.concatenate(feat_data_list, axis=0)
    feat_out_all = np.concatenate(feat_out_list, axis=0)
    return feat_data_all, feat_out_all

def load_one_OD(array, day, hour):
    feat_out_list = []
    feat_data_list = []

    feat_out = array[day, hour]
    mylen = np.zeros([len(feat_out), len(feat_out)])
    # if (feat_out == mylen).all():
    #     print("+1")
    # else:
    #     print("-1")
    feat_in = feat_out.T
    feat_o = feat_out.sum(axis=1, keepdims=True)
    feat_i = feat_in.sum(axis=1, keepdims=True)
    feat_data = np.concatenate((feat_out, feat_in, feat_o, feat_i), axis=1)

    feat_out_list.append(feat_out.reshape([1, 268, 268]))
    feat_data_list.append(feat_data.reshape([1, 268, 538]))

    feat_data_all = np.concatenate(feat_data_list, axis=0)
    feat_out_all = np.concatenate(feat_out_list, axis=0)
    return feat_data_all, feat_out_all

def MAE(pred, gt):
    all_loss = abs(pred - gt)
    all_loss = all_loss.sum(axis=2).sum(axis=1).sum(axis=0)
    all_loss = all_loss / pred.shape[2] / pred.shape[1] / pred.shape[0]
    return all_loss


def RMSE(pred, gt):
    all_loss = abs(pred - gt) ** 2
    all_loss = all_loss.sum(axis=2).sum(axis=1).sum(axis=0)
    all_loss = all_loss / pred.shape[2] / pred.shape[1] / pred.shape[0]

    return math.sqrt(all_loss)


def SMAPE(pred, gt):
    count = gt.shape[2] * gt.shape[1] * gt.shape[0]
    all_loss = 2 * abs(pred - gt) / (abs(pred) + abs(gt) + 1)
    all_loss = all_loss.sum(axis=(0, 1, 2))
    all_loss = all_loss / count
    return all_loss


def MAPE(pred, gt):
    count = gt.shape[2] * gt.shape[1] * gt.shape[0]
    all_loss = abs(pred - gt) / (abs(gt) + 1)
    all_loss = all_loss.sum(axis=(0, 1, 2))
    all_loss = all_loss / count
    return all_loss


def PCC(pred, gt):
    pred_s = pred.reshape(pred.shape[2] * pred.shape[1] * pred.shape[0])
    gt_s = gt.reshape(gt.shape[2] * gt.shape[1] * gt.shape[0])
    pccs = np.corrcoef(pred_s, gt_s)
    return pccs[0][1]


def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")


def analysis_result(result, ground):
    print(MAE(result, ground), 'RMSE', RMSE(result, ground), 'PCC', PCC(result, ground), 'SMAPE', SMAPE(result, ground))
    return ',' + str(MAE(result, ground)) + ',' + str(RMSE(result, ground)) + ',' + \
           str(PCC(result, ground)) + ',' + str(SMAPE(result, ground))


def pre_weight(neighbor, m_size):
    mask = Variable(torch.zeros(len(neighbor), m_size))
    for i in range(m_size):
        grid_no = i
        neighs = neighbor[i]
        for neigh in neighs.keys():
            mask[grid_no, neigh] = neighs[neigh]

    num_neigh1 = mask.sum(1, keepdim=True) + 0.00001
    mask = mask.div(num_neigh1)
    '''
    if (mask == 0).all():
        print("no")
    else:
        print("yes")
        '''
    return mask


def pre_weight_geo(neighbor, m_size):
    mask = Variable(torch.zeros(len(neighbor), m_size))
    for i in range(m_size):
        grid_no = i
        neighs = neighbor[i]
        for neigh in neighs.keys():
            mask[grid_no, neigh] = 1 / neighs[neigh]
    num_neigh1 = mask.sum(1, keepdim=True)
    mask = mask.div(num_neigh1)
    return mask


def get_mask_matrix(weights, m_size):
    tmp = Variable(torch.zeros(len(weights), len(weights[0]), len(weights[0])))
    for t in range(len(weights)):
        for i in range(m_size):
            for j in range(m_size):
                tmp[t][i][j] = weights[t][i][j]
    return tmp
