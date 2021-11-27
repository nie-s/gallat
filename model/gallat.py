import os
import time
import torch
import random

import numpy as np
import torch.nn as nn

from datetime import datetime
from torch.nn import init
from torch.autograd import Variable

from utils.utils import load_OD_matrix, analysis_result


class gallat(nn.Module):

    # def __init__(self, enc1, enc2, rnn, m_size, embed_dim):
    def __init__(self, enc1, enc2, device, epochs, random_seed, lr, batch_size, m_size, embed_dim, rnn):
        super(gallat, self).__init__()
        self.enc1 = enc1
        self.enc2 = enc2
        self.MSE = nn.MSELoss()
        self.m_size = m_size
        self.embed_dim = embed_dim

        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.random_seed = random_seed
        self.epochs = epochs

        self.rnn = rnn
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

    def fit(self, dataset, data, epochs, train_day, vali_day, test_day):
        batch_nodes = list(range(self.m_size))
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        times = []

        save_path = 'training/' + dataset + '__' + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
        os.makedirs(save_path, exist_ok=True)

        log_path = save_path + dataset + '.csv'
        fo = open(log_path, "w")
        fo.write("w_in=" + str(gallat.w_in) + "; w_out=" + str(gallat.w_out) + "; w_all=" + str(gallat.w_all))
        fo.write("Learning_rate=" + str(self.lr) + 'Random_seed=' + str(self.random_seed) + "\n")

        fo.write("Epoch, Loss, Loss_in, Loss_out, Loss_OD, TrainMAE, TrainRMSE, TrainPCC, TrainSMAPE, TrainMAPE"
                 "ValiMAE, ValiRMSE, ValiPCC, ValiSMAPE, ValiMAPE"
                 "TestMAE, TestRMSE, TestPCC, TestSMAPE, TestMAPE, TotalTime, AveTime\n")

        for epoch in range(self.epochs):
            one_time = time.time()
            # if epoch % 100 == 0:
            result = []
            ground = []
            start_time = time.time()

            print("###########################################################################################")
            print("Training Process: epoch=", epoch)
            batches = random.sample(range(12, 44), self.batch_size)
            print("Training iterations = ", batches)
            str_batches = map(lambda x: str(x), batches)
            fo.write(str(epoch) + ",")

            loss = torch.zeros(1, device=self.device)
            loss_in = torch.zeros(1, device=self.device)
            loss_out = torch.zeros(1, device=self.device)
            loss_all = torch.zeros(1, device=self.device)

            for n in range(train_day - 1):
                feat_data, feat_out = load_OD_matrix(data, n, batches)
                feat_out = torch.FloatTensor(feat_out).to(device=self.device)
                feat_data = torch.FloatTensor(feat_data).to(device=self.device)

                ground_truth = Variable(torch.FloatTensor(np.array(data[n + 1, batches])).to(device=self.device))
                optimizer.zero_grad()
                loss_one, loss_in_one, loss_out_one, loss_all_one, od_matrix \
                    = self.loss(feat_data, feat_out, batch_nodes, ground_truth)
                loss += loss_one

                loss_in += loss_in_one
                loss_out += loss_out_one
                loss_all += loss_all_one

                if epoch % 100 == 0:
                    result.append(od_matrix.detach().cpu().numpy())
                    ground.append(ground_truth.detach().cpu().numpy())

            param_count = 0
            '''
            for param in geml.parameters():
                l1_regularization += torch.mean(torch.abs(param))
                param_count += 1

            l1_regularization = torch.div(l1_regularization, param_count)
            # loss = loss + l1_regularization
            '''

            print("Loss=", loss.item(), 'Loss_in=', loss_in.item(), 'Loss_out=', loss_out.item(), 'Loss_OD=',
                  loss_all.item())
            fo.write(str(loss.item()) + ',' + str(loss_in.item()) + ',' +
                     str(loss_out.item()) + ',' + str(loss_all.item()))
            loss.backward(retain_graph=True)
            optimizer.step()

            if epoch % 100 == 0:
                result = np.concatenate(result, axis=0).reshape(-1, 268, 268)
                ground = np.concatenate(ground, axis=0).reshape(-1, 268, 268)
                fo.write(analysis_result(result, ground))
            else:
                fo.write("\n")

            # Vali and Test
            if (epoch % 100 == 0):
                # torch.save(agg1.state_dict(), save_path + str(model_no) + '-' + str(epoch) + "agg1.pt")
                # torch.save(agg2.state_dict(), save_path + str(model_no) + '-' + str(epoch) + "agg2.pt")
                torch.save(self.enc1.state_dict(), save_path + dataset + '-' + str(epoch) + "enc1.pt")
                torch.save(self.enc2.state_dict(), save_path + dataset + '-' + str(epoch) + "enc2.pt")
                torch.save(self.state_dict(), save_path + dataset + '-' + str(epoch) + "gallat.pt")

                #  Testing Process
                result_vali = []
                ground_vali = []
                for day in range(train_day, train_day + vali_day):
                    halfHours = [hour for hour in range(12, 44)]
                    feat_data, feat_out = load_OD_matrix(data, day, halfHours)
                    feat_out = torch.FloatTensor(feat_out).to(device=self.device)
                    feat_data = torch.FloatTensor(feat_data).to(device=self.device)

                    _, ground_truth = load_OD_matrix(data, day + 1, halfHours)

                    od_matrix, _, _ = self.forward(feat_data, feat_out, batch_nodes)
                    result_vali.append(od_matrix.detach().cpu().numpy())
                    ground_vali.append(ground_truth)

                result_vali = np.concatenate(result, axis=0).reshape(-1, 268, 268)
                ground_vali = np.concatenate(ground, axis=0).reshape(-1, 268, 268)
                print('Vali')
                fo.write(analysis_result(result_vali, ground_vali))

                result = []
                ground = []

                for day in range(train_day + vali_day, train_day + vali_day + test_day):
                    halfHours = [hour for hour in range(12, 44)]
                    feat_data, feat_out = load_OD_matrix(data, day, halfHours)
                    feat_out = torch.FloatTensor(feat_out).to(device=self.device)
                    feat_data = torch.FloatTensor(feat_data).to(device=self.device)

                    _, ground_truth = load_OD_matrix(data, day + 1, halfHours)

                    od_matrix, _, _ = self.forward(feat_data, feat_out, batch_nodes)
                    result.append(od_matrix.detach().cpu().numpy())
                    ground.append(ground_truth)

                result = np.concatenate(result, axis=0).reshape(-1, 268, 268)
                ground = np.concatenate(ground, axis=0).reshape(-1, 268, 268)
                np.save(save_path + dataset + '-' + str(epoch), result)

                print('Test')
                fo.write(analysis_result(result, ground))
                times.append(time.time() - start_time)

                print("Total time of training:", np.sum(times))
                print("Average time of training:", np.mean(times) / 100)
                fo.write(',' + str(np.sum(times)) + ',' + str(np.mean(times)) + '\n')
            print('one_epoch_time', time.time() - one_time)
        fo.close()
