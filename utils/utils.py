import numpy as np


def getGraph(graph_path):
    return np.load(graph_path)


def load_OD_matrix(array, day, hours):
    feat_out_list = []
    feat_data_list = []
    for hour in hours:
        feat_out = array[day, hour]
        feat_in = feat_out.T
        feat_o = feat_out.sum(axis=1, keepdims=True)
        feat_i = feat_in.sum(axis=1, keepdims=True)
        feat_data = np.concatenate((feat_out, feat_in, feat_o, feat_i), axis=1)

        feat_out_list.append(feat_out.reshape([1, 268, 268]))
        feat_data_list.append(feat_data.reshape([1, 268, 538]))

    feat_data_all = np.concatenate(feat_data_list, axis=0)
    feat_out_all = np.concatenate(feat_out_list, axis=0)
    return feat_data_all, feat_out_all
