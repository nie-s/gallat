import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class temporal_attention(nn.Module):

    def __init__(self, feature_dim, embed_dim, device):
        super(temporal_attention, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.device = device

        self.wq_1 = nn.Parameter(torch.zeros(size=(feature_dim, embed_dim)))
        self.wk_1 = nn.Parameter(torch.zeros(size=(embed_dim, embed_dim)))
        self.wv_1 = nn.Parameter(torch.zeros(size=(embed_dim, embed_dim)))

        self.wq_2 = nn.Parameter(torch.zeros(size=(feature_dim, embed_dim)))
        self.wk_2 = nn.Parameter(torch.zeros(size=(embed_dim, embed_dim)))
        self.wv_2 = nn.Parameter(torch.zeros(size=(embed_dim, embed_dim)))

        self.wq_3 = nn.Parameter(torch.zeros(size=(feature_dim, embed_dim)))
        self.wk_3 = nn.Parameter(torch.zeros(size=(embed_dim, embed_dim)))
        self.wv_3 = nn.Parameter(torch.zeros(size=(embed_dim, embed_dim)))

        self.wq_4 = nn.Parameter(torch.zeros(size=(feature_dim, embed_dim)))
        self.wk_4 = nn.Parameter(torch.zeros(size=(embed_dim, embed_dim)))
        self.wv_4 = nn.Parameter(torch.zeros(size=(embed_dim, embed_dim)))

        self.wq = nn.Parameter(torch.zeros(size=(feature_dim, embed_dim)))
        self.wk = nn.Parameter(torch.zeros(size=(embed_dim, embed_dim)))
        self.wv = nn.Parameter(torch.zeros(size=(embed_dim, embed_dim))) # todo 这些全初始化成全0可能不太行吧

    def forward(self, features, s1, s2, s3, s4):
        ms1 = cal(features, s1, self.wq_1, self.wk_1, self.wv_1, self.embed_dim, self.device)
        ms2 = cal(features, s2, self.wq_2, self.wk_2, self.wv_2, self.embed_dim, self.device)
        ms3 = cal(features, s3, self.wq_3, self.wk_3, self.wv_3, self.embed_dim, self.device)
        ms4 = cal(features, s4, self.wq_4, self.wk_4, self.wv_4, self.embed_dim, self.device)
        ms = torch.zeros(size=(268, self.embed_dim), device=self.device)

        for x in (ms1, ms2, ms3, ms4):
            query = torch.mm(torch.tensor(features, dtype=torch.float32, device=self.device), self.wq)
            key = torch.mm(x, self.wk_4)
            value = torch.mm(x, self.wv_4)
            kq = torch.mul(query, key)
            kq = F.softmax(torch.div(kq, math.sqrt(self.embed_dim)))
            ms = torch.add(ms, torch.mul(kq, value))

        return ms


def cal(features, s, wq, wk, wv, embed_dim, device):
    ms = torch.zeros(size=(268, embed_dim), device=device)
    for i in range(s.shape[0]):
        mt = s[i].to(device=device)
        query = torch.mm(torch.tensor(features, dtype=torch.float32, device=device), wq)
        key = torch.mm(mt, wk)
        value = torch.mm(mt, wv)
        kq = torch.mul(query, key)
        kq = F.softmax(torch.div(kq, math.sqrt(embed_dim)))
        ms = torch.add(ms, torch.mul(kq, value))

    return ms
