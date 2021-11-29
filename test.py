import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from model.attention_net import attention_net
from torch.autograd import Variable

from utils.utils import get_graph, load_geo_neighbors, load_OD_matrix, name_with_datetime, load_backward_neighbors, \
    load_forward_neighbors

from model.spatial import spatial_attention


def pre_weight(neighbor, nodes):
    mask = Variable(torch.zeros(len(neighbor), len(nodes)))
    for i in range(len(nodes)):
        grid_no = i
        neighs = neighbor[i]
        for neigh in neighs.keys():
            mask[grid_no, neigh] = neighs[neigh]
    num_neigh1 = mask.sum(1, keepdim=True) + 0.00001
    mask = mask.div(num_neigh1)
    return mask


batches = random.sample(range(12, 44), 20)

data_path = '/home/zengjy/data/Beijing/'

data = np.load(data_path + 'all.npy')
graph = get_graph(data_path + 'graph.npy')


def test_attention():
    # feat_data = 20 x 268 x 538  batch_size x m_size x d
    # feat_out = 20 x 268 x 268   batch_size x m_size x m_size
    feat_data, feat_out = load_OD_matrix(data, 0, batches)
    forward_adj, forward_neighbors = load_forward_neighbors(feat_out, m_size=268, batch_num=0)
    backward_adj, backward_neighbors = load_backward_neighbors(feat_out, m_size=268, batch_num=0)
    geo_neighbors = load_geo_neighbors(graph, m_size=268, geo_thr=3)

    attention_forward = attention_net(538, 20, 0.2)
    attention_backward = attention_net(538, 20, 0.2)
    attention_geo = attention_net(538, 20, 0.2)

    batch_nodes = list(range(268))

    # t1 = torch.tensor(pre_weight(forward_neighbors, batch_nodes), dtype=torch.float32)
    # t2 = torch.tensor(feat_data[0], dtype=torch.float32)
    # mask_forward = torch.mm(t1, t2)
    #
    # t1 = torch.tensor(pre_weight(backward_neighbors, batch_nodes), dtype=torch.float32)
    # t2 = torch.tensor(feat_data[0], dtype=torch.float32)
    # mask_backward = torch.mm(t1, t2)
    #
    # t1 = torch.tensor(pre_weight(geo_neighbors, batch_nodes), dtype=torch.float32)
    # t2 = torch.tensor(feat_data[0], dtype=torch.float32)
    # mask_geo = torch.mm(t1, t2)
    #
    # attention_forward.forward(feat_data[0], mask_forward, forward_adj)
    # attention_backward.forward(feat_data[0], mask_backward, backward_adj)
    # attention_geo.forward(feat_data[0], mask_geo, graph)

    # def __init__(self, nnode, feature_dim, embed_dim, device, geo_adj, forward_adj, backward_adj,
    #              geo_neighbors, forward_neighbors, backward_neighbors):

    spatial = spatial_attention(268, 538, 20, 'cuda:0', graph, forward_adj, backward_adj, geo_neighbors,
                                forward_neighbors, backward_neighbors)

    spatial.forward(feat_data[0])


test_attention()

history_m = defaultdict(dict)

