import torch
import os.path as osp
import pandas as pd
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, to_undirected
import numpy as np
import json

def get_raw_text_instagram(use_text=False, seed=0):
    if osp.exists(f"./datasets/instagram/instagram.pt"):
        data = torch.load(f"./datasets/instagram/instagram.pt", map_location='cpu')
        # data.x = data.x.float() # Half into Float
        edge_index = to_undirected(data.edge_index)
        # edge_index, _ = add_self_loops(data.edge_index)
        data.edge_index = edge_index
        data.num_nodes = data.y.shape[0]

        # split data
        node_id = np.arange(data.num_nodes)
        np.random.shuffle(node_id)

        data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
        data.val_id = np.sort(
            node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
        data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

        data.train_mask = torch.tensor(
            [x in data.train_id for x in range(data.num_nodes)])
        data.val_mask = torch.tensor(
            [x in data.val_id for x in range(data.num_nodes)])
        data.test_mask = torch.tensor(
            [x in data.test_id for x in range(data.num_nodes)])
        raw_texts_np = np.array(data.raw_texts)  # 转为 NumPy 数组
        raw_texts_np[raw_texts_np == ""] = "None"  # 向量化替换
        data.raw_texts = raw_texts_np.tolist()  # 转回列表（如需）
        with open('./datasets/instagram.json', 'w', encoding='utf-8') as f:
            json.dump(data.raw_texts, f, ensure_ascii=False, indent=2)
        return data, data.raw_texts
    else:
        raise NotImplementedError('No existing instagram dataset!')