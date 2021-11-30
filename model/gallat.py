import os
import time
import torch
import random

import numpy as np
import torch.nn as nn

from datetime import datetime
from torch.nn import init
from torch.autograd import Variable

from model.spatial import spatial_attention
from model.temporal import temporal_attention
from model.transferring import transferring_attention
from utils.utils import load_OD_matrix, analysis_result, load_geo_neighbors, name_with_datetime, \
    load_backward_neighbors, load_forward_neighbors


class gallat(nn.Module):

    def __init__(self, device, epochs, random_seed, lr, batch_size, m_size, feature_dim, embed_dim, batch_no, time_slot,
                 graph):
        super(gallat, self).__init__()

        self.smooth_loss = nn.SmoothL1Loss()

        self.m_size = m_size
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.batch_no = batch_no

        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.random_seed = random_seed
        self.epochs = epochs
        self.time_slot = time_slot

        # loss weight
        self.wd = 0.8
        self.wo = 0.2
        self.graph = graph

        self.tran_Matrix = nn.Parameter(torch.FloatTensor(self.embed_dim, self.embed_dim))

    def forward(self, features, features_1, feat_out, history_spatial_embedding, day, hour):

        forward_adj, forward_neighbors = load_forward_neighbors(feat_out, m_size=self.m_size)
        backward_adj, backward_neighbors = load_backward_neighbors(feat_out, m_size=self.m_size)
        geo_neighbors = load_geo_neighbors(self.graph, m_size=self.m_size, geo_thr=3)

        spatial = spatial_attention(self.m_size, self.feature_dim, self.embed_dim, self.device)
        spatial_embedding = spatial.forward(features, self.graph, forward_adj, backward_adj, geo_neighbors,
                                            forward_neighbors, backward_neighbors)

        history_spatial_embedding[day][hour] = spatial_embedding

        s1, s2, s3, s4 = get_history_embedding(day, hour, history_spatial_embedding, 'bj', self.time_slot)
        temporal = temporal_attention(self.feature_dim, 4 * self.embed_dim)
        mt = temporal.forward(features_1, s1, s2, s3, s4)

        transferring = transferring_attention(self.m_size, 4 * self.embed_dim, 8 * self.embed_dim, self.device)
        demand, od_matrix = transferring.forward(mt)

        return od_matrix, demand, history_spatial_embedding

    def loss(self, features, features_t1, feat_out, ground_truth, history_spatial_embeddings, day, hour):
        od_matrix, demand, spatial_embedding = self.forward(features, features_t1, feat_out, history_spatial_embeddings,
                                                            day, hour)

        gt_out = torch.div(ground_truth.sum(1, keepdim=True), self.m_size)

        loss_d = torch.mul(self.smooth_loss(demand, gt_out), self.wd)
        loss_o = torch.mul(self.smooth_loss(od_matrix, ground_truth), self.wo)
        loss = loss_d + loss_o

        return loss, loss_d, loss_o, od_matrix, spatial_embedding

    def vali_test(self, data, start, end, halfHours, history_spatial_embeddings):
        result = []
        ground = []

        embed = history_spatial_embeddings

        for n in range(start, end):
            feat_data, feat_out = load_OD_matrix(data, n, halfHours)
            feat_out = torch.FloatTensor(feat_out)
            feat_data = torch.FloatTensor(feat_data)

            for m in range(30):
                ground_truth = Variable(torch.FloatTensor(np.array(data[n, m + 1])))

                od_matrix, demand, spatial_embedding = self.forward(feat_data[m], feat_data[m + 1], feat_out[m],
                                                                    embed, n, m)

                embed = spatial_embedding

                result.append(od_matrix.detach().cpu().numpy())
                ground.append(ground_truth.detach().cpu().numpy())

        result = np.concatenate(result, axis=0).reshape(-1, self.m_size, self.m_size)
        ground = np.concatenate(ground, axis=0).reshape(-1, self.m_size, self.m_size)

        return result, ground, history_spatial_embeddings

    def fit(self, dataset, data, epochs, train_day, vali_day, test_day):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        times = []

        save_path = 'training/' + dataset + '__' + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
        os.makedirs(save_path, exist_ok=True)

        log_path = save_path + dataset + '.csv'
        fo = open(log_path, "w")
        fo.write("w_d=" + str(self.wd) + "; w_o=" + str(self.wo) + "; ")
        fo.write("Learning_rate=" + str(self.lr) + 'Random_seed=' + str(self.random_seed) + "\n")

        fo.write("Epoch, Loss, Loss_d, Loss_o, TrainMAE, TrainRMSE, TrainPCC, TrainSMAPE, TrainMAPE"
                 "ValiMAE, ValiRMSE, ValiPCC, ValiSMAPE, ValiMAPE"
                 "TestMAE, TestRMSE, TestPCC, TestSMAPE, TestMAPE, TotalTime, AveTime\n")

        for epoch in range(self.epochs):
            one_time = time.time()

            result = []
            ground = []
            start_time = time.time()

            print("###########################################################################################")
            print("Training Process: epoch=", epoch)
            halfHours = [hour for hour in range(12, 45)]
            fo.write(str(epoch) + ",")

            # loss = torch.zeros(1, device=self.device)
            # loss_d = torch.zeros(1, device=self.device)
            # loss_o = torch.zeros(1, device=self.device)
            loss = torch.zeros(1)
            loss_d = torch.zeros(1)
            loss_o = torch.zeros(1)

            history_spatial_embeddings = torch.FloatTensor(train_day + vali_day + test_day, self.batch_no, self.m_size,
                                                           4 * self.embed_dim)

            for n in range(train_day - 1):
                feat_data, feat_out = load_OD_matrix(data, n, halfHours)
                feat_out = torch.FloatTensor(feat_out)
                feat_data = torch.FloatTensor(feat_data)

                for m in range(30):
                    ground_truth = Variable(torch.FloatTensor(np.array(data[n, m + 1])))
                    optimizer.zero_grad()
                    loss_one, loss_d_one, loss_o_one, od_matrix, spatial_embedding = \
                        self.loss(feat_data[m], feat_data[m + 1], feat_out[m], ground_truth, history_spatial_embeddings,
                                  n, m)
                    history_spatial_embeddings = spatial_embedding
                    loss += loss_one
                    loss_d += loss_d_one
                    loss_o += loss_o_one

                    result.append(od_matrix.detach().cpu().numpy())
                    ground.append(ground_truth.detach().cpu().numpy())

            print("Loss=", loss.item(), 'Loss_d=', loss_d.item(), 'Loss_o=', loss_o.item())
            fo.write(str(loss.item()) + ',' + str(loss_d.item()) + ',' + str(loss_o.item()) + ',')
            loss.backward(retain_graph=True)
            optimizer.step()

            fo.write("\n")
            result = np.concatenate(result, axis=0).reshape(-1, self.m_size, self.m_size)
            ground = np.concatenate(ground, axis=0).reshape(-1, self.m_size, self.m_size)
            fo.write(analysis_result(result, ground))

            torch.save(self.state_dict(), save_path + dataset + '-' + str(epoch) + "gallat.pt")

            #  Validation Process
            result_vali, ground_vali, embedding = self.vali_test(data, train_day, train_day + vali_day, halfHours,
                                                                 history_spatial_embeddings)

            history_spatial_embeddings = embedding

            print('Vali')
            fo.write(analysis_result(result_vali, ground_vali))

            # Testing Process
            result, ground, embedding = self.vali_test(data, train_day + vali_day, train_day + vali_day + test_day,
                                                       halfHours,
                                                       history_spatial_embeddings)

            np.save(save_path + dataset + '-' + str(epoch), result)

            print('Test')
            fo.write(analysis_result(result, ground))
            times.append(time.time() - start_time)

            print("Total time of training:", np.sum(times))
            print("Average time of training:", np.mean(times) / 100)
            fo.write(',' + str(np.sum(times)) + ',' + str(np.mean(times)) + '\n')
            print('one_epoch_time', time.time() - one_time)

        fo.close()


def get_history_embedding(day, hour, history, dataset, time_slot):
    day_len = min(day, time_slot)
    hour_len = max(0, hour - time_slot + 1)

    s1 = []
    for i in range(day_len):
        s1.append(history[day - i][hour + 1])

    s2 = []
    for i in range(day_len):
        s2.append(history[day - i][hour])

    s3 = []
    for i in range(day_len):
        s3.append(history[day - i][hour + 2])

    s4 = []
    for i in range(hour_len, hour + 1):
        s4.append(history[day][i])

    s1 = torch.tensor([item.detach().numpy() for item in s1])
    s2 = torch.tensor([item.detach().numpy() for item in s2])
    s3 = torch.tensor([item.detach().numpy() for item in s3])
    s4 = torch.tensor([item.detach().numpy() for item in s4])

    print(s4.shape)

    # val= torch.tensor([item.cpu().detach().numpy() for item in val]).cuda()

    return s1, s2, s3, s4
