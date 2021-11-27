import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

W = [[2, 1, 4, 6, 1], [3, 1, 8, 9, 5]]

v_i = torch.tensor([1, 2, 3, 4, 5])
v_i = torch.unsqueeze(v_i, 0)

v_j = torch.tensor([10, 20, 30, 40, 50])
v_j = torch.unsqueeze(v_j, 0)

h_i = torch.mm(torch.tensor(W), torch.tensor(v_i).T)
h_j = torch.mm(torch.tensor(W), torch.tensor(v_j).T)

a = [[12, 54, 64, 13]]
N = h_i.size()[0]

print(h_i.repeat(1, N).view(N * N, -1))
print(h_j.repeat(N, 1))
print(torch.cat([h_i.repeat(1, N).view(N * N, -1), h_j.repeat(N, 1)], dim=1))
a_input = torch.cat([h_i.repeat(1, N).view(N * N, -1), h_j.repeat(N, 1)], dim=1) \
    .view(N, -1, 2 * 4)

print(a_input)
print("----------------")
le = nn.LeakyReLU(0.2)
e = torch.matmul(a_input, torch.tensor(a).T).squeeze(2)
print(e)
zero_vec = -9e15 * torch.ones_like(e)

# attention = torch.where(adj > 0, e, zero_vec)
attention = F.softmax(e.float(), dim=1)
print(attention)
