from math import sqrt

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class spatial_attention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, feature_dim, embed_dim, geo_neighbors):
        super(spatial_attention, self).__init__()

        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.geo_neighbors = geo_neighbors
        self.weight = nn.Parameter(
            torch.zeros(size=(embed_dim, feature_dim)))
        init.xavier_uniform_(self.weight)

    def forward(self, features, feat_out, nodes):
        """
        Generates embeddings for a batch of nodes.
        """
        geo_neighs = [self.geo_neighbors[int(node)] for node in nodes]

        # Mean Aggregator Entry
        neigh_feats = self.aggregator.forward(nodes, features, feat_out, geo_neighs, self.num_sample)
        if self.gcn == False:
            self_feats = features
            combined = torch.cat([self_feats, neigh_feats], dim=2)
        else:
            combined = neigh_feats

        combined = F.relu(self.weight.matmul(torch.transpose(combined, 1, 2)))

        return torch.transpose(combined, 1, 2)
