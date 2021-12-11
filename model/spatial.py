import torch
import torch.nn as nn
from torch.nn import init
from model.attention_net import attention_net

from utils.utils import pre_weight, pre_weight_geo


class spatial_attention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, m_size, feature_dim, embed_dim, device):
        super(spatial_attention, self).__init__()

        self.m_size = m_size
        self.feature_dim = feature_dim  # d
        self.embed_dim = embed_dim  # de
        self.device = device

        self.attention_geo = attention_net(feature_dim, embed_dim, 2 * embed_dim, 0.2, m_size, device).to(device=device)
        self.attention_forward = self.attention_geo
        self.attention_backward = self.attention_forward

        self.weight = nn.Parameter(
            torch.FloatTensor(size=(embed_dim, feature_dim))).to(device=device)
        init.xavier_uniform_(self.weight)

        self.flag = torch.zeros([100, 100])
        self.neighbor_list = [[0] * 100 for i in range(100)]

    def forward(self, features, geo_adj, forward_adj, backward_adj, geo_neighbors, forward_neighbors,
                backward_neighbors, day, hour):
        t = features
        if self.flag[day][hour] == 0:
            mask_forward = torch.mm(pre_weight(forward_neighbors, self.m_size).to(self.device), t)
            mask_backward = torch.mm(pre_weight(backward_neighbors, self.m_size).to(self.device), t)
            mask_geo = torch.mm(pre_weight_geo(geo_neighbors, self.m_size).to(self.device), t)
            self.flag[day][hour] = 1
            self.neighbor_list[day][hour] = [mask_forward, mask_backward, mask_geo]
        else:
            mask_forward, mask_backward, mask_geo = self.neighbor_list[day][hour]
        # print(mask_forward)
        # print(mask_backward)
        # print((mask_geo == 0).all())
        # print("=======")
        weight_forward = self.attention_forward.forward(features, mask_forward, forward_adj)
        weight_backward = self.attention_backward.forward(features, mask_backward, backward_adj)
        weight_geo = self.attention_geo.forward(features, mask_geo, geo_adj)
        # print(weight_forward)
        # print(weight_backward)
        # print("=======")
        x = torch.mm(self.weight, t.T)

        # zero_vec = -9e15 * torch.ones_like(t, device=self.device)
        t_ = t
        t_expand = t_.reshape(self.feature_dim, 1, self.m_size)
        t_expand.to(self.device)

        x_forward = torch.mul(weight_forward, t_expand).sum(1)  # 不是mm
        x_forward = torch.mm(self.weight, x_forward)
        x_backward = torch.mul(weight_backward, t_expand).sum(1)
        x_backward = torch.mm(self.weight, x_backward)
        x_geo = torch.mul(weight_geo, t_expand).sum(1)
        x_geo = torch.mm(self.weight, x_geo)

        # aggregator

        m = torch.cat([x_forward, x, x_geo, x_backward])  # todo 这里我改了一下顺序 为了查看变量的值

        return m.T
