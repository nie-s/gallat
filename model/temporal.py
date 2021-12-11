import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class temporal_attention(nn.Module):

    def __init__(self, feature_dim, embed_dim, device):
        super(temporal_attention, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.device = device

        self.wq_1 = nn.Parameter(torch.FloatTensor(feature_dim, embed_dim).to(device=device))
        self.wk_1 = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim).to(device=device))
        self.wv_1 = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim).to(device=device))

        self.wq_2 = nn.Parameter(torch.FloatTensor(feature_dim, embed_dim).to(device=device))
        self.wk_2 = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim).to(device=device))
        self.wv_2 = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim).to(device=device))

        self.wq_3 = nn.Parameter(torch.FloatTensor(feature_dim, embed_dim).to(device=device))
        self.wk_3 = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim).to(device=device))
        self.wv_3 = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim).to(device=device))

        self.wq_4 = nn.Parameter(torch.FloatTensor(feature_dim, embed_dim).to(device=device))
        self.wk_4 = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim).to(device=device))
        self.wv_4 = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim).to(device=device))

        self.wq = nn.Parameter(torch.FloatTensor(feature_dim, embed_dim).to(device=device))
        self.wk = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim).to(device=device))
        self.wv = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim).to(device=device))

        init.xavier_uniform_(self.wq_1)
        init.xavier_uniform_(self.wq_2)
        init.xavier_uniform_(self.wq_3)
        init.xavier_uniform_(self.wq_4)
        init.xavier_uniform_(self.wq)

        init.xavier_uniform_(self.wk_1)
        init.xavier_uniform_(self.wk_2)
        init.xavier_uniform_(self.wk_3)
        init.xavier_uniform_(self.wk_4)
        init.xavier_uniform_(self.wk)

        init.xavier_uniform_(self.wv_1)
        init.xavier_uniform_(self.wv_2)
        init.xavier_uniform_(self.wv_3)
        init.xavier_uniform_(self.wv_4)
        init.xavier_uniform_(self.wv)

    def forward(self, features, s1, s2, s3, s4):
        ms1 = cal(features, s1, self.wq_1, self.wk_1, self.wv_1, self.embed_dim, self.device)
        # print("====m1====")
        # print(ms1)
        ms2 = cal(features, s2, self.wq_2, self.wk_2, self.wv_2, self.embed_dim, self.device)
        # print("====m2====")
        # print(ms2)
        ms3 = cal(features, s3, self.wq_3, self.wk_3, self.wv_3, self.embed_dim, self.device)
        # print("====m3====")
        # print(ms3)
        ms4 = cal(features, s4, self.wq_4, self.wk_4, self.wv_4, self.embed_dim, self.device)
        # print("====m4====")
        # print(ms4)
        ms = torch.zeros(size=(268, self.embed_dim), device=self.device)
        # print(s1.shape)

        for x in (ms1, ms2, ms3, ms4):
            # print("+1")

            query = torch.mm(features, self.wq)
            key = torch.mm(x, self.wk)  # todo 这里之前似乎写错了
            value = torch.mm(x, self.wv)
            # print(x)
            # print(query)
            # print(key)
            # print(value)
            # print(query.shape)
            # print(key.shape)
            # print(self.embed_dim)
            kq = torch.mm(query, key.T)
            # print(kq)
            temp = torch.div(kq, math.sqrt(self.embed_dim)) # 这里确实不需要*2
            # print(temp)
            kq = F.softmax(temp, dim=1)
            # print(kq)
            temp2 = torch.mm(kq, value)
            ms = torch.add(ms, temp2)

        # print("====ms====")
        # print(ms)
        return ms


def cal(features, s, wq, wk, wv, embed_dim, device):
    ms = torch.zeros(size=(268, embed_dim), device=device)
    for i in range(s.shape[0]):
        mt = s[i].to(device=device)
        # print("=======================mt======================")
        # print(mt)
        # print(mt.shape)
        # print(wk.shape)
        query = torch.mm(features, wq)
        key = torch.mm(mt, wk)
        # print(query)
        # print(key)
        value = torch.mm(mt, wv)
        # print(value)
        kq = torch.mm(query, key.T)  # todo 这里应该还是 matmul
        # print(kq)
        temp = torch.div(kq, math.sqrt(embed_dim))
        # print(temp)
        gg = F.softmax(temp, dim=1)
        # print(gg)
        temp2 = torch.mm(gg, value)
        # print(temp2)
        ms = torch.add(ms, temp2)
        # print(ms)

    return ms
