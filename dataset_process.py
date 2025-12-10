
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
import numpy as np
import torch
import pandas as pd
import random
import os
import json
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid

def get_cora_casestudy(SEED=0):
    data_X, data_Y, data_citeid, data_edges = parse_cora()
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")
    # print("data_X",data_X)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'datasets')
    dataset = Planetoid('./datasets', data_name)
    data = dataset[0]
    # print("data.x",data.x)
    # data_X_tor = torch.from_numpy(data_X)
    # print(torch.allclose(data_X_tor,data.x))
    # df_x = pd.DataFrame(data_X)
    # df_xt = pd.DataFrame(data.x.numpy())
    # with pd.ExcelWriter("save_x.xlsx",engine='openpyxl') as writer:
    #     df_x.to_excel(writer,sheet_name='data_x',index=False)
    #     df_xt.to_excel(writer,sheet_name='data_xt',index=False)
    #
    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    # data.train_id = np.sort(node_id[:int(data.num_nodes * 0.1)])
    # data.val_id = np.sort(
    #     node_id[int(data.num_nodes * 0.1):int(data.num_nodes * 0.2)])
    # data.test_id = np.sort(node_id[int(data.num_nodes * 0.2):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data, data_citeid

def parse_cora():
    path = 'datasets/cora_orig/cora'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    # print("idx_features_labels:",idx_features_labels)
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    # print("data_X:",data_X)
    # print("labels:",labels)
    # print("data_Y:",data_Y)
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()
#
# path = './datasets/cora_orig/cora'
# idx_features_labels = np.genfromtxt("{}.content".format(path), dtype=np.dtype(str))
# data_citeid = idx_features_labels[:, 0]
#
# # 读取论文文件名映射
# with open('./datasets/cora_orig/mccallum/cora/papers') as f:
#     lines = f.readlines()
# pid_filename = {}
# for line in lines:
#     pid = line.split('\t')[0]
#     fn = line.split('\t')[1]
#     if fn == 'http:##www.cs.ucc.ie#~dgb#papers#ICCBR2.ps.Z':
#         fn = 'http:##www.cs.ucc.ie#~dgb#papers#iccbr2.ps.Z'
#     if fn == 'http:##www.cs.ucl.ac.uk#staff#t.yu#ep97.ps':
#         fn = 'http:##www.cs.ucl.ac.uk#staff#T.Yu#ep97.ps'
#     if fn == 'http:##www.cs.ucl.ac.uk#staff#t.yu#pgp.new.ps':
#         fn = 'http:##www.cs.ucl.ac.uk#staff#T.Yu#pgp.new.ps'
#     pid_filename[pid] = fn
#
# # 加载文本
# path = './datasets/cora_orig/mccallum/cora/extractions/'
# text = []
# all_text = {}
# for pid in data_citeid:
#     fn = pid_filename[pid]
#     with open(path + fn) as f:
#         lines = f.read().splitlines()
#     all_text[pid] = lines
#     ti, ab = 'Title: None', 'Abstract: None'
#     found_title, found_abstract = False, False
#     for line in lines:
#         if not found_title and line.strip().startswith('Title:'):
#             ti = line
#             found_title = True
#         if not found_abstract and line.strip().startswith('Abstract:'):
#             ab = line
#             found_abstract = True
#         if found_title and found_abstract:
#             break
#     if ab == 'Abstract: None':
#         text.append(ti)
#     elif ti == 'Title: None':
#         text.append(ab)
#     else:
#         text.append(ti + '\n' + ab)
# with open('./datasets/cora_ori_text.json', 'w', encoding='utf-8') as outf:
#     json.dump(all_text,outf,ensure_ascii=False,indent=2)
# with open('./datasets/cora_text.json', 'w', encoding='utf-8') as f:
#     json.dump(text, f, ensure_ascii=False, indent=2)
# with open("./r_assign.txt","r") as f:
#     new_labels = [int(line.strip()) for line in f.readlines()]
#
# datasets = Dataset.from_list([
#     {"text": t, "label": int(label)} for t, label in zip(text, new_labels)
# ])
# split_dataset = datasets.train_test_split(test_size=0.1, seed=0)
# dataset_dict = DatasetDict({
#     'train': split_dataset['train'],
#     'test': split_dataset['test']
# })
# dataset_dict.save_to_disk('./text_datasets/cora_after_comm')
