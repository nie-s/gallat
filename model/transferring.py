import torch
import torch.nn as nn
from model.attention_net import attention_net


class transferring_attention(nn.Module):

    def __init__(self, m_size, feature_dim, embed_dim, device):
        super(transferring_attention, self).__init__()

        self.m_size = m_size
        self.feature_dim = feature_dim  # d
        self.embed_dim = embed_dim  # de
        self.device = device

        self.attention_net = attention_net(feature_dim, feature_dim, embed_dim, 0.2, m_size, device=device)

        self.wd = nn.Linear(feature_dim, 1, bias=True)

        self.act = torch.nn.ReLU()

    def forward(self, mt):
        q = self.attention_net.forward(mt, mt, torch.ones(self.m_size, self.m_size))  # n x n
        # print(q)
        d = self.act(self.wd(mt))
        # print(q)
        # print(d)
        q = torch.mul(q, d)
        '''
        for i in range(self.m_size):
            q[i] = torch.mul(q[i],
                             d[i][0])
        '''
        # print(q)
        return d, q
