import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class attention_net(nn.Module):
    def __init__(self, feature_dim, embed_dim, a_dim, alpha, m_size, device):
        super(attention_net, self).__init__()

        self.feature_dim = feature_dim  # d
        self.embed_dim = embed_dim  # de
        self.a_dim = a_dim
        self.alpha = alpha
        self.device = device
        self.m_size = m_size

        self.W = nn.Parameter(torch.zeros(size=(feature_dim, embed_dim))).to(device=device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(a_dim, 1))).to(device=device)  # d x N
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyRelu = nn.LeakyReLU(self.alpha).to(device=device)

    def forward(self, v_i, v_j, adj):
        # v : N x d                  n 4de
        # w : N x de x d             n 4de 4de
        # print("=============v============")
        # print(v_i)
        h_i = torch.mm(v_i, self.W)  # N x de
        h_j = torch.mm(v_j, self.W)  # N x de

        a_input = torch.cat([
            h_i.repeat(1, self.m_size).view(self.m_size * self.m_size, h_i.shape[1]),
            h_j.repeat(self.m_size, 1)],
            dim=1)  # N*N x 2de

        # print(a_input.shape)

        e = self.leakyRelu(torch.mm(a_input, self.a)).reshape(self.m_size, -1)  # N x N
        # print("e=================")
        # print(e)
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(torch.FloatTensor(adj).to(device=self.device) > 0, e, zero_vec)
        # print(attention)
        attention = F.softmax(attention, dim=1)
        # print(attention)
        return attention
