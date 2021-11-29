import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class attention_net(nn.Module):
    def __init__(self, feature_in, feature_out, alpha, nnode):
        super(attention_net, self).__init__()

        self.feature_in = feature_in  # d
        self.feature_out = feature_out  # de
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(feature_out, feature_in)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * feature_out, nnode)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyRelu = nn.LeakyReLU(self.alpha)

    def forward(self, v_i, v_j, adj):
        # v : N x d
        # w : de x d
        h_i = torch.mm(torch.tensor(v_i, dtype=torch.float32), self.W.T)  # N x de
        h_j = torch.mm(torch.tensor(v_j, dtype=torch.float32), self.W.T)  # N x de

        N = h_i.size()[0]

        a_input = torch.cat([h_i, h_j], dim=1)  # N x 2de

        e = self.leakyRelu(torch.mm(a_input, self.a))  # batch x N x N
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(torch.tensor(adj) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=0)

        return attention
