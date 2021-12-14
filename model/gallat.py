import os
import time
import torch

import numpy as np
import torch.nn as nn

from datetime import datetime
from torch.autograd import Variable
from torch.nn import init

from model.spatial import spatial_attention
from model.temporal import temporal_attention
from model.transferring import transferring_attention
from utils.utils import load_OD_matrix, analysis_result, load_geo_neighbors, load_backward_neighbors, \
    load_forward_neighbors
from tqdm import trange


class gallat(nn.Module):

    def __init__(self, device, epochs, random_seed, lr, batch_size, m_size, feature_dim, embed_dim, batch_no, time_slot,
                 graph, data, start_hour, end_hour, data_torch, neighbor_list, mask_list):
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

        self.spatial_attention = spatial_attention(self.m_size, self.feature_dim, self.embed_dim, self.device,
                                                   mask_list).to(
            device=device)
        self.temporal_attention = temporal_attention(self.feature_dim, 4 * self.embed_dim, self.device).to(
            device=device)
        self.transferring_attention = transferring_attention(self.m_size, 4 * self.embed_dim, 8 * self.embed_dim,
                                                             self.device, ).to(device=device)

        # loss weight
        self.wd = 0.8
        self.wo = 0.2
        self.graph = graph
        self.tran_Matrix = nn.Parameter(torch.FloatTensor(self.embed_dim, self.embed_dim)).to(device=device)
        init.xavier_uniform_(self.tran_Matrix)

        self.data = data
        self.data_torch = data_torch
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.features = torch.eye(self.m_size, device=device)
        self.mask_list = mask_list

        self.neighbor_list = neighbor_list

    def get_spatial_embedding(self, day, hour):
        st = time.time()
        forward_adj, forward_neighbors, backward_adj, backward_neighbors, geo_neighbors = \
            self.neighbor_list[day][hour]

        # print(time.time() - st)
        st = time.time()
        spatial_embedding = self.spatial_attention.forward(self.features, self.graph, forward_adj, backward_adj,
                                                           geo_neighbors,
                                                           forward_neighbors, backward_neighbors, day=day, hour=hour)
        # print(time.time() - st)
        # print("==========")
        return spatial_embedding.reshape([1, len(spatial_embedding), -1])

    def forward(self, day, hour):
        st = time.time()
        s1, s2, s3, s4 = self.get_history_embedding(day, hour, self.time_slot, self.device)
        # print(time.time() - st)
        st = time.time()
        mt = self.temporal_attention.forward(self.features, s1, s2, s3, s4)
        # print("====mt====")
        # print(mt)
        # print(time.time() - st)
        st = time.time()
        demand, od_matrix = self.transferring_attention.forward(mt)
        # print(od_matrix)
        # print(time.time() - st)

        return od_matrix, demand

    def loss(self, day, hour):
        ground_truth = torch.FloatTensor(np.array(self.data[day, hour + 1 + self.start_hour]))

        od_matrix, demand = self.forward(day, hour)
        demand = demand.squeeze(1)
        gt_out = ground_truth.sum(dim=1)
        # print(demand)
        # print(gt_out)
        loss_d = torch.mul(
            self.smooth_loss(demand.to(device=self.device), gt_out.to(device=self.device)).to(device=self.device),
            self.wd)
        loss_o = torch.mul(self.smooth_loss(od_matrix.to(device=self.device), ground_truth.to(device=self.device)),
                           self.wo)
        # print(od_matrix)
        # print(ground_truth)
        # print(loss_d)

        # print("==============")
        # print(gt_out)
        # print(demand)
        # print(loss_o)
        # print("==============")
        loss = loss_d + loss_o

        return loss, loss_d, loss_o, od_matrix, ground_truth

    def vali_test(self, data, start, end, halfHours):
        result = []
        ground = []

        batch_range = trange(0, (end - start) * halfHours)  # todo 之前这里为啥要-1?
        for _ in batch_range:
            n = (start + _) % halfHours
            m = _ % halfHours
            nx = n
            ny = self.start_hour + m + 1
            if ny == 48:
                ny = 0
                nx = n + 1
            ground_truth = torch.FloatTensor(np.array(data[nx, ny]))

            od_matrix, demand = self.forward(n, m)

            result.append(od_matrix.detach().cpu().numpy())
            ground.append(ground_truth.detach().cpu().numpy())

        result = np.concatenate(result, axis=0).reshape(-1, self.m_size, self.m_size)
        ground = np.concatenate(ground, axis=0).reshape(-1, self.m_size, self.m_size)

        return result, ground

    def fit(self, dataset, data, epochs, train_day, vali_day, test_day, start_hour, end_hour, data_torch):
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
            torch.autograd.set_detect_anomaly(True)
            print("###########################################################################################")
            print("Training Process: epoch=", epoch)
            halfHours = (end_hour - start_hour + 1)
            # result_vali, ground_vali = self.vali_test(data, train_day, train_day + vali_day, halfHours)
            # fo.write(str(epoch) + ",")
            result_vali, ground_vali = self.vali_test(data, train_day, train_day + vali_day, halfHours)
            batch_range = trange(500, train_day * halfHours)  # todo 之前这里为啥要-1?
            for _ in batch_range:
                n = _ // halfHours
                m = _ % halfHours

                optimizer.zero_grad()
                loss_one, loss_d_one, loss_o_one, od_matrix, ground_truth = self.loss(n, m)
                # print(od_matrix)

                loss = loss_one
                loss_d = loss_d_one
                loss_o = loss_o_one

                result.append(od_matrix.detach().cpu().numpy())
                ground.append(ground_truth.detach().cpu().numpy())

                batch_range.set_description(f"train_loss: {loss_one};")
                loss.backward(retain_graph=True)
                optimizer.step()

            print("Loss=", loss.item(), 'Loss_d=', loss_d.item(), 'Loss_o=', loss_o.item())
            # fo.write(str(loss.item()) + ',' + str(loss_d.item()) + ',' + str(loss_o.item()) + ',')

            # fo.write("\n")
            result = np.concatenate(result, axis=0).reshape(-1, self.m_size, self.m_size)
            ground = np.concatenate(ground, axis=0).reshape(-1, self.m_size, self.m_size)
            fo.write(analysis_result(result, ground))

            torch.save(self.state_dict(), save_path + dataset + '-' + str(epoch) + "gallat.pt")

            #  Validation Process
            result_vali, ground_vali = self.vali_test(data, train_day, train_day + vali_day, halfHours)

            print('Vali')
            fo.write(analysis_result(result_vali, ground_vali))

            # Testing Process
            result, ground = self.vali_test(data, train_day + vali_day, train_day + vali_day + test_day, halfHours)

            np.save(save_path + dataset + '-' + str(epoch), result)

            print('Test')
            fo.write(analysis_result(result, ground))
            times.append(time.time() - start_time)

            print("Total time of training:", np.sum(times))
            print("Average time of training:", np.mean(times) / 100)
            fo.write(',' + str(np.sum(times)) + ',' + str(np.mean(times)) + '\n')
            print('one_epoch_time', time.time() - one_time)

        fo.close()

    def get_history_embedding(self, day, hour, time_slot, device):
        day_len = min(day, time_slot)
        hour_len = max(0, hour - time_slot + 1)

        s1 = torch.tensor([]).to(device=device)
        for i in range(day_len):
            s1 = torch.cat([s1, self.get_spatial_embedding(day - i, hour + 1)])

        s2 = torch.tensor([]).to(device=device)
        for i in range(day_len):
            s2 = torch.cat([s2, self.get_spatial_embedding(day - i, hour)])

        s3 = torch.tensor([]).to(device=device)
        for i in range(day_len):
            s3 = torch.cat([s3, self.get_spatial_embedding(day - i, hour + 2)])

        s4 = torch.tensor([]).to(device=device)
        for i in range(hour_len, hour + 1):
            s4 = torch.cat([s4, self.get_spatial_embedding(day, i)])
        # print("=====s1======")
        # print(s1)
        # print("=====s2======")
        # print(s2)
        # print("=====s3======")
        # print(s3)
        # print("=====s4======")
        # print(s4)
        return s1, s2, s3, s4
