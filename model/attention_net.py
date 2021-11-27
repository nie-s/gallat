import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class attention_net(nn.Module):
    def __init__(self, feature_in, feature_out, alpha, geo_neighbors, forward_neighbors, backward_neighbors):
        super(attention_net, self).__init__()

        self.feature_in = feature_in
        self.feature_out = feature_out
        self.alpha = alpha
        self.geo_neighbors = geo_neighbors

        self.W = nn.Parameter(torch.zeros(size=(feature_in, feature_out)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * feature_in, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyRelu = nn.LeakyReLU(self.alpha)

    def forward(self, v_i, v_j, adj):
        h_i = torch.mm(self.W, np.array(v_i).T)
        h_j = torch.mm(self.W, np.array(v_j).T)
        N = h_i.size()[0]

        a_input = torch.cat([h_i.repeat(1, N).view(N * N, -1), h_j.repeat(N, 1)], dim=1) \
            .view(N, -1, 2 * self.feature_out)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        return attention
