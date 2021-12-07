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

        self.W = nn.Parameter(torch.zeros(size=(m_size, embed_dim, feature_dim))).to(device=device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(a_dim, m_size))).to(device=device)  # d x N
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyRelu = nn.LeakyReLU(self.alpha).to(device=device)

    def forward(self, v_i, v_j, adj):
        # v : N x d                  n 4de
        # w : N x de x d             n 4de 4de
        v = v_i.unsqueeze(2).to(device=self.device)
        vv = v_j.unsqueeze(2).to(device=self.device)
        h_i = torch.matmul(self.W, v).squeeze()  # N x de
        h_j = torch.matmul(self.W, vv).squeeze()  # N x de
        a_input = torch.cat([h_i, h_j], dim=1)  # N x 2de

        e = self.leakyRelu(torch.mm(self.a.T, a_input.T))  # N x N
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(torch.FloatTensor(adj).to(device=self.device) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=0)

        return attention
