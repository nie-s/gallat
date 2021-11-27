import torch
import torch.nn as nn
from torch.nn import init


class gallat(nn.Module):

    def __init__(self, enc1, enc2, rnn, m_size, embed_dim):
        super(gallat, self).__init__()
        self.enc1 = enc1
        self.enc2 = enc2
        self.rnn = rnn
        self.MSE = nn.MSELoss()
        self.m_size = m_size
        self.embed_dim = embed_dim
        self.w_in = 0.25
        self.w_out = 0.25
        self.w_all = 0.5
        self.tran_Matrix = nn.Parameter(torch.FloatTensor(self.embed_dim, self.embed_dim))
        init.xavier_uniform_(self.tran_Matrix)  # pytorch的初始化函数
        self.tran_Matrix_in = nn.Parameter(torch.FloatTensor(1, self.embed_dim))
        init.xavier_uniform_(self.tran_Matrix_in)
        self.tran_Matrix_out = nn.Parameter(torch.FloatTensor(1, self.embed_dim))
        init.xavier_uniform_(self.tran_Matrix_out)
        self.hn = nn.Parameter(torch.FloatTensor(1, self.m_size, self.embed_dim))
        init.xavier_uniform_(self.hn)
        self.cn = nn.Parameter(torch.FloatTensor(1, self.m_size, self.embed_dim))
        init.xavier_uniform_(self.cn)

    def forward(self, features, feat_out, nodes):
        mid_features = self.enc1(features, feat_out, nodes)
        embeds = self.enc2(mid_features, feat_out, nodes)
        inputs = embeds.reshape(32, self.m_size, self.embed_dim)
        output, (hn, cn) = self.rnn(inputs, (self.hn, self.cn))
        self.hn = nn.Parameter(hn)
        self.cn = nn.Parameter(cn)
        output = output.reshape(32, self.m_size, self.embed_dim)
        od_matrix = output.matmul(self.tran_Matrix).matmul(torch.transpose(output, 1, 2)).float()
        od_in = torch.div(output.matmul(self.tran_Matrix_in.t()).float(), self.m_size)
        od_out = torch.div(output.matmul(self.tran_Matrix_out.t()).float(), self.m_size)
        return od_matrix, od_out, od_in

    def loss(self, features, feat_out, nodes, ground_truth):
        od_matrix, od_out, od_in = self.forward(features, feat_out, nodes)
        gt_out = torch.div(ground_truth.sum(2, keepdim=True), self.m_size)
        gt_in = torch.div(torch.transpose(ground_truth.sum(1, keepdim=True), 1, 2), self.m_size)
        loss_in = torch.mul(self.MSE(od_in, gt_in), self.w_in)
        loss_out = torch.mul(self.MSE(od_out, gt_out), self.w_out)
        loss_all = torch.mul(self.MSE(od_matrix, ground_truth), self.w_all)
        loss = loss_in + loss_out + loss_all
        return loss, loss_in, loss_out, loss_all, od_matrix
