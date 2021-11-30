import torch
import torch.nn as nn
from torch.nn import init
from model.attention_net import attention_net
from utils.utils import pre_weight


class transferring_attention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, m_size, feature_dim, embed_dim, device):
        super(transferring_attention, self).__init__()

        self.m_size = m_size
        self.feature_dim = feature_dim  # d
        self.embed_dim = embed_dim  # de
        self.device = device

        self.attention_net = attention_net(feature_dim, feature_dim, embed_dim, 0.2, m_size)

        self.wd = nn.Linear(feature_dim, 1, bias=True)

    def forward(self, mt):
        q = self.attention_net.forward(mt, mt, torch.ones(self.m_size, self.m_size))
        d = torch.sigmoid(self.wd(mt))


        for i in range(self.m_size):
            q[i].mul(d[i])

        return d, q
