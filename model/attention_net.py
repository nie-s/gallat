import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class attention_net(nn.Module):
    def __init__(self, feature_dim, embed_dim, geo_neighbors):
        super(attention_net, self).__init__()

        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.geo_neighbors = geo_neighbors
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, feature_dim))
        init.xavier_uniform_(self.weight)
