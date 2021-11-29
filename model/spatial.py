from math import sqrt

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from model.attention_net import attention_net

from utils.utils import pre_weight, get_mask_matrix_i, get_mask_matrix_j


class spatial_attention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, nnode, feature_dim, embed_dim, device, geo_adj, forward_adj, backward_adj,
                 geo_neighbors, forward_neighbors, backward_neighbors):
        super(spatial_attention, self).__init__()

        self.nnode = nnode
        self.feature_dim = feature_dim  # d
        self.embed_dim = embed_dim  # de
        self.device = device

        self.attention_geo = attention_net(feature_dim, embed_dim, 0.2)
        self.attention_forward = attention_net(feature_dim, embed_dim, 0.2)
        self.attention_backward = attention_net(feature_dim, embed_dim, 0.2)

        self.geo_neighbors = geo_neighbors
        self.forward_neighbors = forward_neighbors
        self.backward_neighbors = backward_neighbors

        self.geo_adj = geo_adj
        self.forward_adj = forward_adj
        self.backward_adj = backward_adj

        self.weight = nn.Parameter(
            torch.zeros(size=(embed_dim, feature_dim)))
        init.xavier_uniform_(self.weight)

    def forward(self, features):
        """
        Generates embeddings for a batch of nodes.
        """
        t = torch.tensor(features, dtype=torch.float32)

        batch_nodes = list(range(self.nnode))

        mask_forward = torch.mm(torch.tensor(pre_weight(self.forward_neighbors, batch_nodes), dtype=torch.float32),
                                t)
        mask_backward = torch.mm(
            torch.tensor(pre_weight(self.backward_neighbors, batch_nodes), dtype=torch.float32), t)
        mask_geo = torch.mm(torch.tensor(pre_weight(self.geo_neighbors, batch_nodes)), t)

        weight_forward = self.attention_forward.forward(features, mask_forward, self.forward_adj)
        weight_backward = self.attention_backward.forward(features, mask_backward, self.backward_adj)
        weight_geo = self.attention_geo.forward(features, mask_geo, self.geo_adj)

        x = torch.mm(self.weight, t.T)

        zero_vec = -9e15 * torch.ones_like(t)

        t_expand = t.clone().reshape(self.feature_dim, 1, self.nnode)
        weight_forward_masked = get_mask_matrix_i(weight_forward, self.nnode)
        weight_backward_masked = get_mask_matrix_j(weight_backward, self.nnode)
        weight_geoward_masked = get_mask_matrix_i(weight_geo, self.nnode)

        # attention_forward = torch.where(torch.tensor(self.forward_adj) > 0, t, zero_vec)
        x_forward = torch.mul(weight_forward_masked, t_expand).sum(1)  # 不是mm
        x_forward = torch.mm(self.weight, x_forward)
        x_backward = torch.mul(weight_backward_masked, t_expand).sum(1)
        x_backward = torch.mm(self.weight, x_backward)
        x_geo = torch.mul(weight_geoward_masked, t_expand).sum(1)
        x_geo = torch.mm(self.weight, x_geo)

        m = torch.cat([x, x_forward, x_backward, x_geo])

        return m
