import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dgl
import dgl.data.utils as U
import time
import pickle
from models.layers import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import copy
from sklearn.metrics import precision_score,f1_score,recall_score
import warnings
warnings.filterwarnings('ignore')

class UnircaDataset():
    """
    参数
    ----------
    dataset_path: str
        数据存放位置。
        举例: 'train_Xs.pkl' （67 * 14 * 40）（图数 * 节点数 * 节点向量维数）
    labels_path: str
        标签存放位置。
        举例: 'train_ys_anomaly_type.pkl' （67）
    topology: str
        图的拓扑结构存放位置
        举例：'topology.pkl'
    aug: boolean (default: False)
        需要数据增强，该值设置为True
    aug_size: int (default: 0)
        数据增强时，每个label对应的样本数
    shuffle: boolean (default: False)
        load()完成以后，若shuffle为True，则打乱self.graphs 和 self.labels （同步）
    """

    def __init__(self, dataset_path, labels_path, topology, aug=False, aug_size=0, shuffle=False):
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.topology = topology
        self.aug = aug
        self.aug_size = aug_size
        self.graphs = []
        self.labels = []
        self.load()
        if shuffle:
            self.shuffle()

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def load(self):
        """ __init__()  中使用，作用是装载 self.graphs 和 self.labels，若aug为True，则进行数据增强操作。
        """
        Xs = tensor(U.load_info(self.dataset_path))
        ys = tensor(U.load_info(self.labels_path))
        topology = U.load_info(self.topology)
        assert Xs.shape[0] == ys.shape[0]
        if self.aug:
            Xs, ys = self.aug_data(Xs, ys)

        for X in Xs:
            g = dgl.graph(topology)  # 同质图
            # 若有0入度节点，给这些节点加自环
            in_degrees = g.in_degrees()
            zero_indegree_nodes = [i for i in range(len(in_degrees)) if in_degrees[i].item() == 0]
            for node in zero_indegree_nodes:
                g.add_edges(node, node)

            g.ndata['attr'] = X
            self.graphs.append(g)
        self.labels = ys

    def shuffle(self):
        graphs_labels = [(g, l) for g, l in zip(self.graphs, self.labels)]
        random.shuffle(graphs_labels)
        self.graphs = [i[0] for i in graphs_labels]
        self.labels = [i[1] for i in graphs_labels]

    def aug_data(self, Xs, ys):
        """ load() 中使用，作用是数据增强
        参数
        ----------
        Xs: tensor
            多个图对应的特征向量矩阵。
            举例：67个图对应的Xs规模为 67 * 14 * 40 （67个图，每个图14个节点）
        ys: tensor
            每个图对应的label，要求是从0开始的整数。
            举例：如果一共有10个label，那么ys中元素值为 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        self.aug_size: int
            数据增强时，每个label对应的样本数

        返回值
        ----------
        aug_Xs: tensor
            数据增强的结果
        aug_ys: tensor
            数据增强的结果
        """
        aug_Xs = []
        aug_ys = []
        num_label = len(set([y.item() for y in ys]))
        grouped_Xs = [[] for i in range(num_label)]
        for X, y in zip(Xs, ys):
            grouped_Xs[y.item()].append(X)
        for group_idx in range(len(grouped_Xs)):
            cur_Xs = grouped_Xs[group_idx]
            n = len(cur_Xs)
            m = Xs.shape[1]
            while len(cur_Xs) < self.aug_size:
                select = np.random.choice(n, m)
                aug_X = torch.zeros_like(Xs[0])
                for i, j in zip(select, range(m)):
                    aug_X[j] = cur_Xs[i][j].detach().clone()
                cur_Xs.append(aug_X)
            for X in cur_Xs:
                aug_Xs.append(X)
                aug_ys.append(group_idx)
        aug_Xs = torch.stack(aug_Xs, 0)
        aug_ys = tensor(aug_ys)
        return aug_Xs, aug_ys


class RawDataProcess():
    """用来处理原始数据的类
    参数
    ----------
    config: dict
        配置参数
        Xs: 多个图的特征向量矩阵
        data_dir: 数据和结果存放路径
        dataset: 数据集名称 可选['21aiops', 'gaia']
    """

    def __init__(self, config):
        self.config = config

    def process(self):
        """ 用来获取并保存中间数据
        输入：
            sentence_embedding.pkl
            demo.csv
        输出：
            训练集：
                train_Xs.pkl
                train_ys_anomaly_type.pkl
                train_ys_service.pkl
            测试集：
                test_Xs.pkl
                test_ys_anomaly_type.pkl
                test_ys_service.pkl
            拓扑：
                topology.pkl
        """
        run_table = pd.read_csv(os.path.join(self.config['data_dir'], self.config['run_table']), index_col=0)
        Xs = U.load_info(os.path.join(self.config['data_dir'], self.config['Xs']))
        Xs = np.array(Xs)
        label_types = ['anomaly_type', 'service']
        label_dict = {label_type: None for label_type in label_types}
        for label_type in label_types:
            label_dict[label_type] = self.get_label(label_type, run_table)

        save_dir = self.config['save_dir']
#         train_size = self.config['train_size']
        train_index = np.where(run_table['data_type'].values=='train')
        test_index = np.where(run_table['data_type'].values=='test')
        train_size = len(train_index[0])
        # 保存特征向量，特征向量是先训练集后测试集
#         print(train_index)
        U.save_info(os.path.join(save_dir, 'train_Xs.pkl'), Xs[: train_size])
        U.save_info(os.path.join(save_dir, 'test_Xs.pkl'), Xs[train_size: ])
        # 保存标签
        for label_type, labels in label_dict.items():
            U.save_info(os.path.join(save_dir, f'train_ys_{label_type}.pkl'), labels[train_index])
            U.save_info(os.path.join(save_dir, f'test_ys_{label_type}.pkl'), labels[test_index])
        # 保存拓扑
        topology = self.get_topology()
        U.save_info(os.path.join(save_dir, 'topology.pkl'), topology)
        # 保存边的类型(异质图)
        if self.config['heterogeneous']:
            edge_types = self.get_edge_types()
            U.save_info(os.path.join(save_dir, 'edge_types.pkl'), edge_types)

    def get_label(self, label_type, run_table):
        """ process() 中调用，用来获取label
        参数
        ----------
        label_type: str
            label的类型，可选：['service', 'anomaly_type']
        run_table: pd.DataFrame

        返回值
        ----------
        labels: torch.tensor()
            label列表
        """
        meta_labels = sorted(list(set(list(run_table[label_type]))))
        labels_idx = {label: idx for label, idx in zip(meta_labels, range(len(meta_labels)))}
        labels = np.array(run_table[label_type].apply(lambda label_str: labels_idx[label_str]))
        return labels

    def get_topology(self):
        """ process() 中调用，用来获取topology
        """
        dataset = self.config['dataset']
        if self.config['heterogeneous']:
            # 异质图
            if dataset == 'gaia':
                topology = (
                [8, 6, 8, 4, 6, 4, 2, 9, 1, 3, 3, 7, 1, 7, 5, 0, 8, 8, 9, 9, 8, 8, 9, 9, 8, 8, 9, 9, 2, 2, 3, 3, 0, 0,
                 1, 1, 4, 4, 5, 5, 2, 2, 3, 3, 6, 7, 6, 7, 4, 5, 4, 5, 2, 3, 2, 3, 0, 1, 0, 1, 6, 7, 6, 7, 6, 7, 6, 7,
                 6, 7, 6, 7],
                [6, 8, 4, 8, 4, 6, 9, 2, 3, 1, 7, 3, 7, 1, 0, 5, 6, 7, 6, 7, 4, 5, 4, 5, 2, 3, 2, 3, 0, 1, 0, 1, 6, 7,
                 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 8, 8, 9, 9, 8, 8, 9, 9, 8, 8, 9, 9, 2, 2, 3, 3, 0, 0, 1, 1, 4, 4, 5, 5,
                 2, 2, 3, 3])
            elif dataset == '20aiops':
                topology = (
                [2, 3, 4, 5, 6, 7, 8, 9, 13, 10, 11, 12, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6,
                 7, 8, 9, 13, 13, 10, 10, 11, 11, 12, 12, 1, 6, 7, 8, 9, 1, 6, 7, 8, 9, 1, 6, 7, 8, 9, 1, 6, 7, 8, 9, 0,
                 0, 0, 0, 4, 5, 2, 6, 3, 7, 5, 9],
                [2, 3, 4, 5, 6, 7, 8, 9, 13, 10, 11, 12, 1, 6, 7, 8, 9, 1, 6, 7, 8, 9, 1, 6, 7, 8, 9, 1, 6, 7, 8, 9, 0,
                 0, 0, 0, 4, 5, 2, 6, 3, 7, 5, 9, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 7, 8,
                 9, 13, 13, 10, 10, 11, 11, 12, 12])
            elif dataset == '21aiops':
                topology = (
                    [12, 12, 13, 13, 0, 0, 0, 0, 1, 1, 1, 1, 8, 8, 9, 9, 10, 10, 11, 11, 8, 8, 9, 9, 10, 10, 11, 11, 
                     2, 2, 2, 2, 3, 3, 3, 3, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 0, 1, 8, 
                     9, 10, 11, 2, 3, 14, 15, 16, 17, 0, 1, 0, 1, 8, 9, 10, 11, 8, 9, 10, 11, 6, 4, 6, 4, 6, 4, 6, 4, 
                     2, 3, 2, 3, 2, 3, 2, 3, 14, 15, 16, 17, 14, 15, 16, 17, 7, 7, 7, 7, 5, 5, 5, 5, 2, 2, 2, 2, 3, 3, 
                     3, 3, 0, 1, 8, 9, 10, 11, 2, 3, 14, 15, 16, 17],
                    [0, 1, 0, 1, 8, 9, 10, 11, 8, 9, 10, 11, 6, 4, 6, 4, 6, 4, 6, 4, 2, 3, 2, 3, 2, 3, 2, 3, 14, 15, 16,
                     17, 14, 15, 16, 17, 7, 7, 7, 7, 5, 5, 5, 5, 2, 2, 2, 2, 3, 3, 3, 3, 0, 1, 8, 9, 10, 11, 2, 3, 14, 15,
                     16, 17, 12, 12, 13, 13, 0, 0, 0, 0, 1, 1, 1, 1, 8, 8, 9, 9, 10, 10, 11, 11, 8, 8, 9, 9, 10, 10, 11, 11,
                     2, 2, 2, 2, 3, 3, 3, 3, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 0, 1, 8, 9, 10,
                     11, 2, 3, 14, 15, 16, 17]
                )
            else:
                raise Exception()
        else:
            # 同质图
            if dataset == 'gaia':
                topology = (
                    [8, 6, 8, 4, 9, 2, 0, 5, 3, 1, 3, 7, 1, 7, 6, 4, 8, 8, 9, 9, 8, 8, 9, 9, 8, 8, 9, 9, 2, 2, 3, 3, 0,
                     0,
                     1, 1, 4, 4, 5, 5, 2, 2, 3, 3],
                    [6, 8, 4, 8, 2, 9, 5, 0, 1, 3, 7, 3, 7, 1, 4, 6, 6, 7, 6, 7, 4, 5, 4, 5, 2, 3, 2, 3, 0, 1, 0, 1, 6,
                     7,
                     6, 7, 6, 7, 6, 7, 6, 7, 6, 7])  # 正向
            #                 topology = ([8, 6, 8, 4, 6, 4, 2, 9, 1, 3, 3, 7, 1, 7, 5, 0, 8, 8, 9, 9, 8, 8, 9, 9, 8, 8, 9, 9, 2, 2, 3, 3, 0, 0, 1, 1, 4, 4, 5, 5, 2, 2, 3, 3, 6, 7, 6, 7, 4, 5, 4, 5, 2, 3, 2, 3, 0, 1, 0, 1, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7],
            #                            [6, 8, 4, 8, 4, 6, 9, 2, 3, 1, 7, 3, 7, 1, 0, 5, 6, 7, 6, 7, 4, 5, 4, 5, 2, 3, 2, 3, 0, 1, 0, 1, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 8, 8, 9, 9, 8, 8, 9, 9, 8, 8, 9, 9, 2, 2, 3, 3, 0, 0, 1, 1, 4, 4, 5, 5, 2, 2, 3, 3])  # 使用异质图
            elif dataset == '20aiops':
                # topology = (
                #     [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 13, 13, 13, 10, 10, 11, 11, 12, 12, 10, 11, 12],
                #     [1, 2, 6, 7, 8, 9, 1, 3, 6, 7, 8, 9, 1, 4, 6, 7, 8, 9, 1, 5, 6, 7, 8, 9, 0, 6, 0, 7, 0, 8, 0, 9, 4, 5, 13, 2, 6, 3, 7, 5, 9, 10, 11, 12])  # 正向
                topology = (
                    [1, 2, 6, 7, 8, 9, 1, 3, 6, 7, 8, 9, 1, 4, 6, 7, 8, 9, 1, 5, 6, 7, 8, 9, 0, 6, 0, 7, 0, 8, 0, 9, 4,
                     5, 13, 2, 6, 3, 7, 5, 9, 10, 11, 12],
                    [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 13,
                     13, 13, 10, 10, 11, 11, 12, 12, 10, 11, 12])  # 反向
            elif dataset == '21aiops':
                topology = ([12, 12, 13, 13, 0, 0, 0, 0, 1, 1, 1, 1, 8, 8, 9, 9, 10, 10, 11, 11, 8, 8, 9, 9, 10, 10, 11, 
                             11, 2, 2, 2, 2, 3, 3, 3, 3, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 
                             0, 1, 8, 9, 10, 11, 2, 3, 14, 15, 16, 17, 12, 13],
                            [0, 1, 0, 1, 8, 9, 10, 11, 8, 9, 10, 11, 6, 4, 6, 4, 6, 4, 6, 4, 2, 3, 2, 3, 2, 3, 2, 3, 14, 
                             15, 16, 17, 14, 15, 16, 17, 7, 7, 7, 7, 5, 5, 5, 5, 2, 2, 2, 2, 3, 3, 3, 3, 0, 1, 8, 9, 10, 
                             11, 2, 3, 14, 15, 16, 17, 12, 13])  # 正向
            else:
                raise Exception()
        return topology

    def get_edge_types(self):
        dataset = self.config['dataset']
        if not self.config['heterogeneous']:
            raise Exception()
        if dataset == 'gaia':
            etype = tensor(np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2]).astype(np.int64))
        elif dataset == '20aiops':
            etype = tensor(np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2]).astype(np.int64))
        elif dataset == '21aiops':
            etype = tensor(np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(np.int64))
        else:
            raise Exception()
        return etype

class UnircaLab():
    def __init__(self, config):
        self.config = config
        instances = config['nodes'].split()
        self.ins_dict = dict(zip(instances, range(len(instances))))
        self.demos = pd.read_csv(os.path.join(self.config['data_dir'], self.config['run_table']), index_col=0)
        if config['dataset'] == 'gaia':
            self.topoinfo = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]}
        elif config['dataset'] == '21aiops':
            self.topoinfo = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9, 10, 11], 5: [12, 13], 6: []}
        elif config['dataset'] == '20aiops':
            self.topoinfo = {0: [0, 1], 1: list(range(2, 10)), 2: list(range(10, 14))}
        else:
            raise Exception('Unknow dataset')

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels

    def save_result(self, save_path, data):
        df = pd.DataFrame(data, columns=['top_k', 'accuracy'])
        df.to_csv(save_path, index=False)
    
    def train(self, dataset, key):
        # def hook(module, input, output):
        #     features.append(output)
        #     return None
        if self.config['seed'] is not None:
            torch.manual_seed(self.config['seed'])
#         print('len train_dataset=', len(dataset))
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], collate_fn=self.collate)
#         device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        # print(device)

        in_dim = dataset.graphs[0].ndata['attr'].shape[1]
#         out_dim = len(set([i.item() for i in dataset.labels]))
        out_dim = self.config[key]
        # hid_dim = (in_dim + out_dim) * 2 // 3
        hid_dim = int(np.sqrt(in_dim*out_dim))
        if self.config['heterogeneous']:
            etype = U.load_info(os.path.join(self.config['save_dir'], 'edge_types.pkl'))
            model = RGCNClassifier(in_dim, hid_dim, out_dim, etype).to(device)  # @ 异质图
#             model = RGCNv2Classifier(in_dim, hid_dim, out_dim, etype).to(device)
            # 钩子函数钩取中间结果
#             for (name, module) in model.named_modules():
#                 print("name: ", name)
#             model.conv2.dropout.register_forward_hook(hook)
        else:
#             model = GCNClassifier(in_dim, hid_dim, out_dim).to(device)  # 同质图
#             model = GATClassifier(in_dim, hid_dim, out_dim, 3).to(device) # GAT
#             model = SAGEClassifier(in_dim, hid_dim, out_dim).to(device) # GraphSAGE
#             model = TAGClassifier(in_dim, hid_dim, out_dim) # TAGConv
#             model = GATv2Classifier(in_dim, hid_dim, out_dim, 3).to(device)
#             model = LinearClassifier(in_dim, hid_dim, out_dim).to(device)
#             model = ChebClassifier(in_dim, hid_dim, out_dim, 2, True).to(device) # ChebConv
            model = TAGClassifier(in_dim, hid_dim, out_dim).to(device)
        print(model)

        opt = torch.optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        losses = []
        model.train()
        for epoch in tqdm(range(self.config['epoch'])):
            epoch_loss = 0
            epoch_cnt = 0
            features = []
            for batched_graph, labels in dataloader:
                batched_graph = batched_graph.to(device)
                labels = labels.to(device)
                feats = batched_graph.ndata['attr'].float()
                logits = model(batched_graph, feats)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().item()
                epoch_cnt += 1
            losses.append(epoch_loss / epoch_cnt)
            if len(losses) > self.config['win_size'] and \
                    abs(losses[-self.config['win_size']] - losses[-1]) < self.config['win_threshold']:
                # 保存钩子函数的中间结果
#                 with open('feature_out.pkl', 'wb') as f:
#                     pickle.dump(features, f)
                break

        # loss曲线
#         plt.plot(range(len(losses)), losses)
#         plt.show()
        return model
    
    def multi_trainv2(self, dataset_ts, dataset_ta, dataset_t3):
        if self.config['seed'] is not None:
            torch.manual_seed(self.config['seed'])
        
        weight = 0.5
        device = 'cpu'

        dataloader_ts = DataLoader(dataset_ts, batch_size=self.config['batch_size'], collate_fn=self.collate)
        dataloader_ta = DataLoader(dataset_ta, batch_size=self.config['batch_size'], collate_fn=self.collate)
        dataloader_t3 = DataLoader(dataset_t3, batch_size=self.config['batch_size'], collate_fn=self.collate)

        in_dim_ts = dataset_ts.graphs[0].ndata['attr'].shape[1]
        out_dim_ts = self.config['N_S']
        hid_dim_ts = (in_dim_ts + out_dim_ts) * 2 // 3
        in_dim_ta = dataset_ta.graphs[0].ndata['attr'].shape[1]
        out_dim_ta = self.config['N_A']
        hid_dim_ta = (in_dim_ta + out_dim_ta) * 2 // 3
        in_dim_t3 = dataset_t3.graphs[0].ndata['attr'].shape[1]
        out_dim_t3 = 2
        hid_dim_t3 = (in_dim_t3 + out_dim_t3) * 2 // 3

        if self.config['heterogeneous']:
            etype = U.load_info(os.path.join(self.config['save_dir'], 'edge_types.pkl'))
            model_ts = RGCNMSL(in_dim_ts, hid_dim_ts, out_dim_ts, etype).to(device)  # @ 异质图
            model_ta = RGCNClassifier(in_dim_ta, hid_dim_ta, out_dim_ta, etype).to(device)
        else:
            model_ts = SGCCClassifier(in_dim_ts, hid_dim_ts, out_dim_ts).to(device)
            model_ta = SGCCClassifier(in_dim_ta, hid_dim_ta, out_dim_ta).to(device)
        print(model_ts)
        print(model_ta)
        
        opt_ts = torch.optim.Adam(model_ts.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        opt_ta = torch.optim.Adam(model_ta.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        losses = []
        model_ts.train()
        model_ta.train()
        
        ts_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ts]
        ta_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ta]
        for epoch in tqdm(range(self.config['epoch'])):
            epoch_loss = 0
            epoch_cnt = 0
            features = []
            for i in range(len(ts_samples)):
                # service
                ts_bg = ts_samples[i][0].to(device)
                ts_labels = ts_samples[i][1].to(device)
                ts_feats = ts_bg.ndata['attr'].float()
                ts_logits = model_ts(ts_bg, ts_feats)
                ts_loss = F.cross_entropy(ts_logits, ts_labels)
                # anomaly_type
                ta_bg = ta_samples[i][0].to(device)
                ta_labels = ta_samples[i][1].to(device)
                ta_feats = ta_bg.ndata['attr'].float()
                ta_logits = model_ta(ta_bg, ta_feats)
                ta_loss = F.cross_entropy(ta_logits, ta_labels)
                
                opt_ts.zero_grad()
                opt_ta.zero_grad()
                
                total_loss = weight*ts_loss+(1-weight)*ta_loss
                total_loss.backward()
                opt_ts.step()
                opt_ta.step()
                epoch_loss += total_loss.detach().item()
                epoch_cnt += 1
            losses.append(epoch_loss / epoch_cnt)
            if len(losses) > self.config['win_size'] and \
                    abs(losses[-self.config['win_size']] - losses[-1]) < self.config['win_threshold']:
                break
        return model_ts, model_ta  

    def multi_train(self, dataset_ts, dataset_ta):
        if self.config['seed'] is not None:
            torch.manual_seed(self.config['seed'])
        weight = 0.5
        device = 'cpu'
        dataloader_ts = DataLoader(dataset_ts, batch_size=self.config['batch_size'], collate_fn=self.collate)
        dataloader_ta = DataLoader(dataset_ta, batch_size=self.config['batch_size'], collate_fn=self.collate)
        in_dim = dataset_ts.graphs[0].ndata['attr'].shape[1]
        hid_dim = in_dim * 2 // 3
        out_dim_ts = self.config['N_S']
        out_dim_ta = self.config['N_A']
        if self.config['heterogeneous']:
            etype = U.load_info(os.path.join(self.config['save_dir'], 'edge_types.pkl'))
            model = RGCNMSL(in_dim, hid_dim, out_dim_ts, out_dim_ta, etype).to(device)  # @ 异质图
        else:
            raise Exception("haven't set")
        print(model)
        
        opt = torch.optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        losses = []
        model.train()
        
        ts_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ts]
        ta_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ta]
        for epoch in tqdm(range(self.config['epoch'])):
            epoch_loss = 0
            epoch_cnt = 0
            features = []
            for i in range(len(ts_samples)):
                # 两个任务输入的拓扑、特征一致
                bg = ts_samples[i][0].to(device) 
                feats = bg.ndata['attr'].float()
                ts_labels = ts_samples[i][1].to(device)
                ta_labels = ta_samples[i][1].to(device)
                
                ts_logits, ta_logits  = model(bg, feats)
                ta_loss = F.cross_entropy(ta_logits, ta_labels)
                ts_loss = F.cross_entropy(ts_logits, ts_labels)
                
                opt.zero_grad()
                
                total_loss = weight*ts_loss+(1-weight)*ta_loss
                total_loss.backward()
                opt.step()
                epoch_loss += total_loss.detach().item()
                epoch_cnt += 1
            losses.append(epoch_loss / epoch_cnt)
            if len(losses) > self.config['win_size'] and \
                    abs(losses[-self.config['win_size']] - losses[-1]) < self.config['win_threshold']:
                break
        return model

    def multi_trainv0(self, dataset_ts, dataset_ta):
        if self.config['seed'] is not None:
            torch.manual_seed(self.config['seed'])
        weight = 0.5
        device = 'cpu'
        dataloader_ts = DataLoader(dataset_ts, batch_size=self.config['batch_size'], collate_fn=self.collate)
        dataloader_ta = DataLoader(dataset_ta, batch_size=self.config['batch_size'], collate_fn=self.collate)
        in_dim_ts = dataset_ts.graphs[0].ndata['attr'].shape[1]
        out_dim_ts = self.config['N_S']
        hid_dim_ts = (in_dim_ts + out_dim_ts) * 2 // 3
        in_dim_ta = dataset_ta.graphs[0].ndata['attr'].shape[1]
        out_dim_ta = self.config['N_A']
        hid_dim_ta = (in_dim_ta + out_dim_ta) * 2 // 3
        if self.config['heterogeneous']:
            etype = U.load_info(os.path.join(self.config['save_dir'], 'edge_types.pkl'))
            model_ts = RGCNClassifier(in_dim_ts, hid_dim_ts, out_dim_ts, etype).to(device)
            model_ta = RGCNClassifier(in_dim_ta, hid_dim_ta, out_dim_ta, etype).to(device)
        else:
            model_ts = TAGClassifier(in_dim_ts, hid_dim_ts, out_dim_ts).to(device)
            model_ta = TAGClassifier(in_dim_ta, hid_dim_ta, out_dim_ta).to(device)
            # model = GCNClassifier(in_dim, hid_dim, out_dim).to(device)  # 同质图
#             model = GATClassifier(in_dim, hid_dim, out_dim, 3).to(device) # GAT
#             model = SAGEClassifier(in_dim, hid_dim, out_dim).to(device) # GraphSAGE
#             model = TAGClassifier(in_dim, hid_dim, out_dim) # TAGConv
#             model = GATv2Classifier(in_dim, hid_dim, out_dim, 3).to(device)
#             model = LinearClassifier(in_dim, hid_dim, out_dim).to(device)
#             model = ChebClassifier(in_dim, hid_dim, out_dim, 2, True).to(device) # ChebConv
            # model = SGCCClassifier(in_dim, hid_dim, out_dim).to(device)
        print(model_ts)
        print(model_ta)
        
        opt_ts = torch.optim.Adam(model_ts.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        opt_ta = torch.optim.Adam(model_ta.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        losses = []
        model_ts.train()
        model_ta.train()
        
        ts_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ts]
        ta_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ta]
        for epoch in tqdm(range(self.config['epoch'])):
            epoch_loss = 0
            epoch_cnt = 0
            features = []
            for i in range(len(ts_samples)):
                # service
                ts_bg = ts_samples[i][0].to(device)
                ts_labels = ts_samples[i][1].to(device)
                ts_feats = ts_bg.ndata['attr'].float()
                ts_logits = model_ts(ts_bg, ts_feats)
                ts_loss = F.cross_entropy(ts_logits, ts_labels)
                # anomaly_type
                ta_bg = ta_samples[i][0].to(device)
                ta_labels = ta_samples[i][1].to(device)
                ta_feats = ta_bg.ndata['attr'].float()
                ta_logits = model_ta(ta_bg, ta_feats)
                ta_loss = F.cross_entropy(ta_logits, ta_labels)
                
                opt_ts.zero_grad()
                opt_ta.zero_grad()
                
                total_loss = weight*ts_loss+(1-weight)*ta_loss
                total_loss.backward()
                opt_ts.step()
                opt_ta.step()
                epoch_loss += total_loss.detach().item()
                epoch_cnt += 1
            losses.append(epoch_loss / epoch_cnt)
            if len(losses) > self.config['win_size'] and \
                    abs(losses[-self.config['win_size']] - losses[-1]) < self.config['win_threshold']:
                break
        return model_ts, model_ta  
        
    def trans_train(self, dataset_src, dataset_target, retrain=False):
        if self.config['seed'] is not None:
            torch.manual_seed(self.config['seed'])
        dataloader_src = DataLoader(dataset_src, batch_size=self.config['batch_size'], collate_fn=self.collate)
        device = 'cpu'
        print(device)

        in_dim = dataset_src.graphs[0].ndata['attr'].shape[1]
        out_dim = self.config['N_A']
#         hid_dim = (in_dim + out_dim) * 2 // 3
        hid_dim = in_dim * 2 // 3
        if self.config['heterogeneous']:
            etype = U.load_info(os.path.join(self.config['save_dir'], 'edge_types.pkl'))
            model_src = RGCNClassifier(in_dim, hid_dim, out_dim, etype).to(device)  # @ 异质图
        else:
            model_src = SGCCClassifier(in_dim, hid_dim, out_dim).to(device)
        print(model_src)

        opt = torch.optim.Adam(model_src.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        losses = []
        model_src.train()
        for epoch in tqdm(range(self.config['epoch'])):
            epoch_loss = 0
            epoch_cnt = 0
            features = []
            for batched_graph, labels in dataloader_src:
                batched_graph = batched_graph.to(device)
                labels = labels.to(device)
                feats = batched_graph.ndata['attr'].float()
                logits = model_src(batched_graph, feats)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().item()
                epoch_cnt += 1
            losses.append(epoch_loss / epoch_cnt)
            if len(losses) > self.config['win_size'] and \
                    abs(losses[-self.config['win_size']] - losses[-1]) < self.config['win_threshold']:
                break
        # 至此源模型训练完成，开始目标模型迁移
        model_target = copy.deepcopy(model_src)
        print('retrain: ', retrain)
        if not retrain: # 是否重新训练模型
            for p in model_target.parameters():
                p.requires_grad = False
        dataloader_target = DataLoader(dataset_target, batch_size=self.config['batch_size'], collate_fn=self.collate)
        in_dim = dataset_target.graphs[0].ndata['attr'].shape[1]
        out_dim = self.config['N_S']
        hid_dim = in_dim * 2 // 3
        # 将最后一层替换为新的全连接层，其余层保留，新添加的层默认requires_grad=True
        model_target.classify = nn.Linear(hid_dim, out_dim)
        print(model_target)
        # 重新训练
        opt = torch.optim.Adam(model_target.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        losses = []
        model_target.train()
        for epoch in tqdm(range(self.config['epoch'])):
            epoch_loss = 0
            epoch_cnt = 0
            features = []
            for batched_graph, labels in dataloader_target:
                batched_graph = batched_graph.to(device)
                labels = labels.to(device)
                feats = batched_graph.ndata['attr'].float()
                logits = model_target(batched_graph, feats)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().item()
                epoch_cnt += 1
            losses.append(epoch_loss / epoch_cnt)
            if len(losses) > self.config['win_size'] and \
                    abs(losses[-self.config['win_size']] - losses[-1]) < self.config['win_threshold']:
                break
        
        return model_target
    
    # 获取训练集和测试集的编码
    def get_embedings(self, model, train_dataset, test_dataset):
        model.eval()
        trainloader = DataLoader(train_dataset, batch_size=len(train_dataset) + 10, collate_fn=self.collate)
        testloader = DataLoader(test_dataset, batch_size=len(test_dataset) + 10, collate_fn=self.collate)
        for batched_graph, labels in trainloader:
            train_embeds = model.get_embeds(batched_graph, batched_graph.ndata['attr'].float())
        
        for batched_graph, labels in testloader:
            test_embeds = model.get_embeds(batched_graph, batched_graph.ndata['attr'].float())
        dataset = self.config['dataset']
        with open(f'results/{dataset}_train_embeds.pkl', 'wb') as f:
            pickle.dump(train_embeds, f)
        with open(f'results/{dataset}_test_embeds.pkl', 'wb') as f:
            pickle.dump(test_embeds, f)
        return
    
    def test_cls(self, model, train_dataset, test_dataset, classifier, task):
        model.eval()
        trainloader = DataLoader(train_dataset, batch_size=len(train_dataset) + 10, collate_fn=self.collate)
        testloader = DataLoader(test_dataset, batch_size=len(test_dataset) + 10, collate_fn=self.collate)
        for batched_graph, labels in trainloader:
            train_embeds = model.get_embeds(batched_graph, batched_graph.ndata['attr'].float(), True)
            classifier.fit(train_embeds.detach().numpy(), labels.detach().numpy())
        
        for batched_graph, labels in testloader:
            test_embeds = model.get_embeds(batched_graph, batched_graph.ndata['attr'].float(), True)
#             score = classifier.score(test_embeds.detach().numpy(), labels.detach().numpy())
#             print('score: ', score)
            output = classifier.predict_proba(test_embeds.detach().numpy())
            labels = labels.detach().numpy().reshape(-1, 1)
            # print(classifier.score(test_embeds.detach().numpy(), labels))
            preds = [
                [
                    item[-1] for item in sorted(list(zip(output[i], range(len(output[i]))))[: 5], reverse=True)
                    ] for i in range(len(output))
                ]
            if task == 'instance':
                ser_res = pd.DataFrame(np.append(preds, labels, axis=1), columns=
                                       np.append([f'Top{i}' for i in range(1, len(preds[0])+1)], 'GroundTruth'))
                self.test_instance_local(ser_res, max_num=2)
            elif task == 'anomaly_type':
                preds = np.array(preds)
                pre = precision_score(labels, preds[:, 0], average='weighted')
                rec = recall_score(labels, preds[:, 0], average='weighted')
                f1 = f1_score(labels, preds[:, 0], average='weighted')
                print('Weighted precision', pre)
                print('Weighted recall', rec)
                print('Weighted f1-score', f1)
            else:
                raise Exception('Unknow task')
            
        return
    
    def testv2(self, model, dataset, task, out_file, save_file=None):
        model.eval()
        dataloader = DataLoader(dataset, batch_size=len(dataset) + 10, collate_fn=self.collate)
        device = 'cpu'
        seed = self.config['seed']
        accuracy = []
        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            output = model(batched_graph, batched_graph.ndata['attr'].float())
            k = 5 if output.shape[-1] >= 5 else output.shape[-1]
            if task == 'instance':
                _, indices = torch.topk(output, k=k, dim=1, largest=True, sorted=True)  
                out_dir = os.path.join(self.config['save_dir'], 'preds')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                y_pred = indices.detach().numpy()
                y_true = labels.detach().numpy().reshape(-1, 1)
                ser_res = pd.DataFrame(np.append(y_pred, y_true, axis=1), 
                                       columns=np.append([f'Top{i}' for i in range(1, len(y_pred[0])+1)], 'GroundTruth'))
                
                # 定位到实例级别
                accs, ins_res = self.test_instance_local(ser_res, max_num=2)
                ins_res.to_csv(f'{out_dir}/multitask_seed{seed}_{out_file}')
                columns = ['A@1', 'A@2', 'A@3', 'A@4', 'A@5']
            elif task == 'anomaly_type':
                _, indices = torch.topk(output, k=k, dim=1, largest=True, sorted=True)  
                out_dir = os.path.join(self.config['save_dir'], 'preds')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                y_pred = indices.detach().numpy()
                y_true = labels.detach().numpy().reshape(-1, 1)
                pre = precision_score(y_pred[:, 0], y_true, average='weighted')
                rec = recall_score(y_pred[:, 0], y_true, average='weighted')
                f1 = f1_score(y_pred[:, 0], y_true, average='weighted')
                print('Weighted precision', pre)
                print('Weighted recall', rec)
                print('Weighted f1-score', f1)
                test_cases = self.demos[self.demos['data_type']=='test']
                pd.DataFrame(np.append(
                    y_pred[:, 0].reshape(-1, 1), y_true, axis=1), columns=[
                                           'Pred', 'GroundTruth'], index=test_cases.index).to_csv(
                                               f'{out_dir}/multitask_seed{seed}_{out_file}')
                columns = ['Precision', 'Recall', 'F1-Score']
                accs = np.array([pre, rec, f1])
            else:
                raise Exception('Unknow task')

        if save_file:
            accuracy = pd.DataFrame(accs.reshape(-1, len(columns)), columns=columns)
            save_dir = os.path.join(self.config['save_dir'], 'evaluations', save_file.split('_')[0])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_result(f'{save_dir}/seed{seed}_{save_file}', accuracy)

        return output, labels
    
    def test(self, model, dataset, out_file, save_file=None):
        model.eval()
        dataloader = DataLoader(dataset, batch_size=len(dataset) + 10, collate_fn=self.collate)
#         device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        seed = self.config['seed']
        accuracy = []
        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            if self.config['heterogeneous']:
                output = model(batched_graph, batched_graph.ndata['attr'].float())
            else:
                output = model(batched_graph, batched_graph.ndata['attr'].float())
            for k in range(1, 6):
                values, indices = torch.topk(output, k=k, dim=1, largest=True, sorted=True)
                # 保存Top5的预测结果
                if k == 5:
                    out_dir = os.path.join(self.config['save_dir'], 'preds')
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    y_pred = indices.detach().numpy()
                    y_true = labels.detach().numpy().reshape(-1, 1)
                    pd.DataFrame(np.append(y_pred, y_true, axis=1), columns=['Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'GroundTruth']).to_csv(f'{out_dir}/seed{seed}_{out_file}')
                num = 0
                for i in range(len(indices)):
                    num += indices[i].eq(labels[i]).sum().item()
                print(f'top{k} acc: ', num / len(indices))
                accuracy.append([k, num / len(indices)])

        if save_file:
            save_dir = os.path.join(self.config['save_dir'], 'evaluations', save_file.split('_')[0])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_result(f'{save_dir}/seed{seed}_{save_file}', accuracy)

        return output, labels

    def test_multitask(self, model, dataset_ts, dataset_ta, out_file, save_file_ts=None, save_file_ta=None):
        model.eval()
        dataloader_ts = DataLoader(dataset_ts, batch_size=len(dataset_ts) + 10, collate_fn=self.collate)
        dataloader_ta = DataLoader(dataset_ta, batch_size=len(dataset_ta) + 10, collate_fn=self.collate)
        ts_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ts]
        ta_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ta]
        device = 'cpu'
        seed = self.config['seed']
        accuracy_ts = []
        accuracy_ta = []
        for i in range(len(ts_samples)):
            batched_graph = ts_samples[i][0].to(device)
            labels_ts = ts_samples[i][1].to(device)
            labels_ta = ta_samples[i][1].to(device)
            output_ts, output_ta = model(batched_graph, batched_graph.ndata['attr'].float())
            print('service')
            for k in range(1, 6):
                _, indices_ts = torch.topk(output_ts, k=k, dim=1, largest=True, sorted=True)
                
                # 保存Top5的根因微服务组定位预测结果---->实例定位结果
                if k == 5:
                    out_dir = os.path.join(self.config['save_dir'], 'preds')
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    y_pred = indices_ts.detach().numpy()
                    y_true = labels_ts.detach().numpy().reshape(-1, 1)
                    # pd.DataFrame(np.append(y_pred, y_true, axis=1), columns=['Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'GroundTruth']).to_csv(f'{out_dir}/multitask_seed{seed}_{out_file}')
                    ser_res = pd.DataFrame(np.append(y_pred, y_true, axis=1), columns=
                                           np.append([f'Top{i}' for i in range(1, len(y_pred[0])+1)], 'GroundTruth'))
                    # 定位到实例级别
                    print('instance')
                    _, ins_res = self.test_instance_local(ser_res, 2)
                    ins_res.to_csv(f'{out_dir}/multitask_seed{seed}_{out_file}')
                      
                # num_ts = 0
                # for i in range(len(indices_ts)):
                #     num_ts += indices_ts[i].eq(labels_ts[i]).sum().item()
                # print(f'top{k} acc: ', num_ts / len(indices_ts))
                # accuracy_ts.append([k, num_ts / len(indices_ts)])
                
            print('anomaly type') # anomaly type需要求pre、rec、f1
            for k in range(1, 6):
                _, indices_ta = torch.topk(output_ta, k=k, dim=1, largest=True, sorted=True)
                num_ta = 0
                for i in range(len(indices_ta)):
                    num_ta += indices_ta[i].eq(labels_ta[i]).sum().item()
                print(f'top{k} acc: ', num_ta / len(indices_ta))
                accuracy_ts.append([k, num_ta / len(indices_ta)])

        # if save_file_ts:
        #     save_dir = os.path.join(self.config['save_dir'], 'evaluations', save_file_ts.split('_')[0])
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     self.save_result(f'{save_dir}/seed{seed}_{save_file_ts}', accuracy_ts)
        # if save_file_ta:
        #     save_dir = os.path.join(self.config['save_dir'], 'evaluations', save_file_ta.split('_')[0])
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     self.save_result(f'{save_dir}/seed{seed}_{save_file_ta}', accuracy_ta)

        return

    def test_instance_local(self, s_preds, max_num=2):
        """
        根据微服务的预测结果预测微服务的根因实例
        """
        with open(self.config['text_path'], 'rb') as f:
            info = pickle.load(f)
        ktype = type(list(info.keys())[0])
        test_cases = self.demos[self.demos['data_type']=='test']
        topks = np.zeros(5)
        ins_preds = []
        i = 0
        for index, row in test_cases.iterrows():
            index = ktype(index)
            num_dict = {}
            for pair in info[index]:
                num_dict[self.ins_dict[pair[0]]] = len(info[index][pair].split())
            s_pred = s_preds.loc[i]
            ins_pred = []
            for col in list(s_preds.columns)[: -1]:
                temp = sorted([(ins_id, num_dict[ins_id]) for ins_id in self.topoinfo[s_pred[col]]],
                              key=lambda x: x[-1], reverse=True)
                # print(self.topoinfo[s_pred[col]], temp)
                ins_pred.extend([item[0] for item in temp[: max_num]])
            ins_preds.append(ins_pred[: 5])
            for k in range(5):
                if ins_pred[k] == self.ins_dict[row['instance']]:
                    topks[k: ] += 1
                    break        
            i += 1
        print('Top1-5: ', topks/len(test_cases))
        y_true = np.array([self.ins_dict[ins] for ins in test_cases['instance'].values]).reshape(-1, 1)
        return topks/len(test_cases), pd.DataFrame(np.append(
            ins_preds, y_true, axis=1), columns=[
                'Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'GroundTruth'], index=test_cases.index)
                
        
    
    def cross_evaluate(self, s_output, s_labels, a_output, a_labels, save_file=None):
        N_S = self.config['N_S']
        N_A = self.config['N_A']
        TOPK_SA = self.config['TOPK_SA']
        # softmax取正（使用笛卡尔积比大小）
        s_values = nn.Softmax(dim=1)(s_output)
        a_values = nn.Softmax(dim=1)(a_output)
        # 获得 K_ * K_的笛卡尔积
        product = []
        for k in range(len(s_values)):
            service_val = s_values[k]
            anomaly_val = a_values[k]
            m = torch.zeros(N_S * N_A).reshape(N_S, N_A)
            for i in range(N_S):
                for j in range(N_A):
                    m[i][j] = service_val[i] * anomaly_val[j]
            product.append(m)
        # 获得每个笛卡尔积矩阵的topk及坐标
        sa_topks = []
        for idx in range(len(product)):
            m = product[idx]
            topk = []
            last_max_val = 1
            for k in range(TOPK_SA):
                cur_max_val = tensor(0)
                x = 0
                y = 0
                for i in range(N_S):
                    for j in range(N_A):
                        if m[i][j] > cur_max_val and m[i][j] < last_max_val:
                            cur_max_val = m[i][j]
                            x = i
                            y = j
                topk.append(((x, y), cur_max_val.item()))
                last_max_val = cur_max_val
            sa_topks.append(topk)

        # 使用笛卡尔积计算分数得到service + anomaly_type 的topk结果
        accuracy = []
        for k in range(1, TOPK_SA + 1):
            num = 0
            for i in range(len(s_labels)):
                label = (s_labels[i].item(), a_labels[i].item())
                predicts = sa_topks[i][:k]
                for predict in predicts:
                    if predict[0] == label:
                        num += 1
                        break
            print(f'top{k} acc: ', num / len(s_labels))
            accuracy.append([k, num / len(s_labels)])
        if save_file:
            seed = self.config['seed']
            save_dir = os.path.join(self.config['save_dir'], 'evaluations', 'service_anomaly')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_result(f'{save_dir}/seed{seed}_{save_file}', accuracy)

    def do_lab(self, lab_id):
        save_dir = os.path.join(self.config['save_dir'], str(lab_id))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.config['save_dir'] = save_dir
        RawDataProcess(self.config).process()
        # 训练
        s = time.time()
        print('train starts at', s)
        
        # 分别训练模型
#         service_model = self.train(UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
#                                                  os.path.join(save_dir, 'train_ys_service.pkl'),
#                                                  os.path.join(save_dir, 'topology.pkl'),
#                                                  aug=self.config['aug'],
#                                                  aug_size=self.config['aug_size'],
#                                                  shuffle=True), 'N_S')
                                                 
#         anomaly_type_model = self.train(UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
#                                                       os.path.join(save_dir, 'train_ys_anomaly_type.pkl'),
#                                                       os.path.join(save_dir, 'topology.pkl'),
#                                                       aug=self.config['aug'],
#                                                       aug_size=self.config['aug_size'],
#                                                       shuffle=True), 'N_A')

#         trans_model = self.trans_train(UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
#                                                       os.path.join(save_dir, 'train_ys_anomaly_type.pkl'),
#                                                       os.path.join(save_dir, 'topology.pkl'),
#                                                       aug=self.config['aug'],
#                                                       aug_size=self.config['aug_size'],
#                                                       shuffle=True), 
#                                       UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
#                                                  os.path.join(save_dir, 'train_ys_service.pkl'),
#                                                  os.path.join(save_dir, 'topology.pkl'),
#                                                  aug=self.config['aug'],
#                                                  aug_size=self.config['aug_size'],
#                                                  shuffle=True),
#                                       retrain=True)
        t1 = time.time()
        print('train ends at', t1)
        model_ts, model_ta = self.multi_trainv0(UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
                                                      os.path.join(save_dir, 'train_ys_service.pkl'),
                                                      os.path.join(save_dir, 'topology.pkl'),
                                                      aug=self.config['aug'],
                                                      aug_size=self.config['aug_size'],
                                                      shuffle=True), 
                                      UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
                                                 os.path.join(save_dir, 'train_ys_anomaly_type.pkl'),
                                                 os.path.join(save_dir, 'topology.pkl'),
                                                 aug=self.config['aug'],
                                                 aug_size=self.config['aug_size'],
                                                 shuffle=True))

        # model_m = self.multi_train(UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
        #                                               os.path.join(save_dir, 'train_ys_service.pkl'),
        #                                               os.path.join(save_dir, 'topology.pkl'),
        #                                               aug=self.config['aug'],
        #                                               aug_size=self.config['aug_size'],
        #                                               shuffle=True), 
        #                               UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
        #                                          os.path.join(save_dir, 'train_ys_anomaly_type.pkl'),
        #                                          os.path.join(save_dir, 'topology.pkl'),
        #                                          aug=self.config['aug'],
        #                                          aug_size=self.config['aug_size'],
        #                                          shuffle=True))
        # self.get_embedings(service_model, UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
        #                                               os.path.join(save_dir, 'train_ys_service.pkl'),
        #                                               os.path.join(save_dir, 'topology.pkl'),
        #                                               aug=self.config['aug'],
        #                                               aug_size=self.config['aug_size']), 
        #                               UnircaDataset(os.path.join(save_dir, 'test_Xs.pkl'),
        #                                              os.path.join(save_dir, 'test_ys_service.pkl'),
        #                                              os.path.join(save_dir, 'topology.pkl')))
        t2 = time.time()
        print('train ends at', t2)
        print('train use time', t1 - s, 's ',t2 - t1, 's')
        # 测试并分析准确率
        s = time.time()
        print('test starts at', s)
#         print('[Training respectively]')
#         print('instance')
        

#         _, _ = self.testv2(service_model,
#                                        UnircaDataset(os.path.join(save_dir, 'test_Xs.pkl'),
#                                                      os.path.join(save_dir, 'test_ys_service.pkl'),
#                                                      os.path.join(save_dir, 'topology.pkl')),
#                                        'instance',
#                                        'service_pred.csv',
#                                        'service_acc.csv')

#         print('anomaly_type')
#         _, _ = self.testv2(anomaly_type_model,
#                                        UnircaDataset(os.path.join(save_dir, 'test_Xs.pkl'),
#                                                      os.path.join(save_dir, 'test_ys_anomaly_type.pkl'),
#                                                      os.path.join(save_dir, 'topology.pkl')),
#                                        'anomaly_type',
#                                        'anomaly_pred.csv',
#                                        'anomaly_type_acc.csv')

#         print('service + anomaly_type:')
#         self.cross_evaluate(s_output, s_labels, a_output, a_labels, 'anomaly_type_service_acc.csv')
        # print('[traditional classifier]')
        # print('instance')
        # self.test_cls(model_ts, UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
        #                                          os.path.join(save_dir, 'train_ys_service.pkl'),
        #                                          os.path.join(save_dir, 'topology.pkl'),
        #                                          aug=self.config['aug'],
        #                                          aug_size=self.config['aug_size'],
        #                                          shuffle=True), UnircaDataset(os.path.join(save_dir, 'test_Xs.pkl'), 
        #                                          os.path.join(save_dir, 'test_ys_service.pkl'), 
        #                                          os.path.join(save_dir, 'topology.pkl')), 
        #                                          RandomForestClassifier(random_state=0),
        #                                          'instance')
        # RandomForestClassifier(random_state=0)
        # GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        # AdaBoostClassifier(n_estimators=100, random_state=0)
        # print('anomaly type')
        # self.test_cls(model_ta, UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
        #                                          os.path.join(save_dir, 'train_ys_anomaly_type.pkl'),
        #                                          os.path.join(save_dir, 'topology.pkl'),
        #                                          aug=self.config['aug'],
        #                                          aug_size=self.config['aug_size'],
        #                                          shuffle=True), UnircaDataset(os.path.join(save_dir, 'test_Xs.pkl'), 
        #                                          os.path.join(save_dir, 'test_ys_anomaly_type.pkl'), 
        #                                          os.path.join(save_dir, 'topology.pkl')), 
        #                                          RandomForestClassifier(random_state=0),
        #                                          'anomaly_type')
        print('[Multi_task learning v0]')
#         t_output, t_labels = self.test(trans_model,
#                                        UnircaDataset(os.path.join(save_dir, 'test_Xs.pkl'),
#                                                      os.path.join(save_dir, 'test_ys_service.pkl'),
#                                                      os.path.join(save_dir, 'topology.pkl')),
#                                        'service_pred_trans.csv',
#                                        'service_acc_trans.csv')
        print('instance')
        _, _ = self.testv2(model_ts,
                                       UnircaDataset(os.path.join(save_dir, 'test_Xs.pkl'),
                                                     os.path.join(save_dir, 'test_ys_service.pkl'),
                                                     os.path.join(save_dir, 'topology.pkl')),
                                       'instance',
                                       'instance_pred_multi_v0.csv',
                                       'instance_acc_multi_v0.csv')
        print('anomaly type')
        _, _ = self.testv2(model_ta,
                                       UnircaDataset(os.path.join(save_dir, 'test_Xs.pkl'),
                                                     os.path.join(save_dir, 'test_ys_anomaly_type.pkl'),
                                                     os.path.join(save_dir, 'topology.pkl')),
                                       'anomaly_type',
                                       'anomaly_pred_multi_v0.csv',
                                       'anomaly_acc_multi_v0.csv')
        
        # print('multi_task learning')
        # self.test_multitask(model_m,  UnircaDataset(os.path.join(save_dir, 'test_Xs.pkl'),
        #                                              os.path.join(save_dir, 'test_ys_service.pkl'),
        #                                              os.path.join(save_dir, 'topology.pkl')), 
        #                                              UnircaDataset(os.path.join(save_dir, 'test_Xs.pkl'),
        #                                              os.path.join(save_dir, 'test_ys_anomaly_type.pkl'),
        #                                              os.path.join(save_dir, 'topology.pkl')), 
        #                                              'service_pred_multi.csv', 
        #                                              'service_acc_multi.csv', 
        #                                              'anomaly_type_acc_multi.csv')

        t = time.time()
        print('test ends at', t)
        print('test use time', t - s, 's')
        # 保存模型
        if self.config['save_model']:
            torch.save(model_ts, os.path.join(save_dir, 'service_model.pt'))
            torch.save(model_ta, os.path.join(save_dir, 'anomaly_type_model.pt'))

