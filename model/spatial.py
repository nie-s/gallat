from math import sqrt

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from model.attention_net import attention_net

from utils.utils import pre_weight, get_mask_matrix, pre_weight_geo


class spatial_attention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, nnode, feature_dim, embed_dim, device):
        super(spatial_attention, self).__init__()

        self.nnode = nnode
        self.feature_dim = feature_dim  # d
        self.embed_dim = embed_dim  # de
        self.device = device

        self.attention_geo = attention_net(feature_dim, embed_dim, 0.2, nnode)
        self.attention_forward = attention_net(feature_dim, embed_dim, 0.2, nnode)
        self.attention_backward = attention_net(feature_dim, embed_dim, 0.2, nnode)

        self.weight = nn.Parameter(
            torch.zeros(size=(embed_dim, feature_dim)))
        init.xavier_uniform_(self.weight)

    def forward(self, features, geo_adj, forward_adj, backward_adj, geo_neighbors, forward_neighbors,
                backward_neighbors):
        t = torch.tensor(features, dtype=torch.float32)

        mask_forward = torch.mm(torch.tensor(pre_weight(forward_neighbors, self.nnode), dtype=torch.float32),
                                t)
        mask_backward = torch.mm(
            torch.tensor(pre_weight(backward_neighbors, self.nnode), dtype=torch.float32), t)
        mask_geo = torch.mm(torch.tensor(pre_weight(geo_neighbors, self.nnode)), t)

        weight_forward = self.attention_forward.forward(features, mask_forward, forward_adj)
        weight_backward = self.attention_backward.forward(features, mask_backward, backward_adj)
        weight_geo = self.attention_geo.forward(features, mask_geo, geo_adj)

        x = torch.mm(self.weight, t.T)

        zero_vec = -9e15 * torch.ones_like(t)

        t_expand = t.clone().reshape(self.feature_dim, 1, self.nnode)

        x_forward = torch.mul(weight_forward, t_expand).sum(1)  # 不是mm
        x_forward = torch.mm(self.weight, x_forward)
        x_backward = torch.mul(weight_backward, t_expand).sum(1)
        x_backward = torch.mm(self.weight, x_backward)
        x_geo = torch.mul(weight_geo, t_expand).sum(1)
        x_geo = torch.mm(self.weight, x_geo)

        m = torch.cat([x, x_forward, x_backward, x_geo])

        return m.T
