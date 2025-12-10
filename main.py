import hdbscan

from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
import torch.nn as nn
from data.load import load_data
import networkx as nx
import numpy as np
import torch
import json
from sklearn import cluster
from sklearn.metrics import davies_bouldin_score, silhouette_score,fowlkes_mallows_score

from GCNtrain import GCN_train, result
from mambatrainer import mamba_train
import Q
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.cluster import DBSCAN
from mamba.model import MambaTextClassification
from transformers import AutoTokenizer, TrainingArguments

from semantic_feat_extra import semantic_feat_extra
from dataset_process import get_cora_casestudy


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch = 21
dataset_name = 'instagram'  # cora 6, citeseer 11, wikics , photo 10
if __name__ == "__main__":
    file = open("result.csv", "a+")
    print(f"{dataset_name}=======================================================", file=file)
    file.close()
    structure_community_node_number = []
    print("=================================原始数据加载=================================")
    data, text, _ = load_data(dataset_name, use_text=True, seed=0)
    # data, _ = get_cora_casestudy()
    ori_feat = data.x
    label = data.y
    edge = data.edge_index
    num_nodes = data.num_nodes
    print("ori_feat shape:", ori_feat.shape)
    print("label:", label)
    print("=================================graph构建=================================")

    if dataset_name == 'photo':
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        # 填充边
        adj[edge[0], edge[1]] = 1.0
        # 如果是无向图，还需要对称
        adj[edge[1], edge[0]] = 1.0
        A = adj.numpy()
        graph = nx.from_numpy_array(A)
    elif dataset_name == 'cora' or 'citeseer' or 'wikics' or 'pubmed' or 'arxiv_2023' or 'instagram':
        adj = to_scipy_sparse_matrix(edge, num_nodes=num_nodes)
        adj = adj.tocsr()
        adj = adj - sp.diags(adj.diagonal())
        graph = nx.from_scipy_sparse_array(adj)



    print("=================================结构社团筛选=================================")
    # louvain算法获得社团划分
    structure_community = nx.community.louvain_communities(graph, resolution=0.3, threshold=1e-09,
                                                           seed=123)  # res=0.3/0.5/0.8

    # 计算每个社团的节点数量和总共的节点数
    for i in structure_community:
        community_size = len(i)
        structure_community_node_number.append(community_size)
    print("louvain_community_node_number: ", structure_community_node_number)
    print("louvain_community_number: ", len(structure_community))

    # 计算均值,标准差
    mean_size = np.mean(structure_community_node_number)
    std_deviation = np.std(structure_community_node_number)
    threshold = mean_size + 0.5 * std_deviation
    print("mean_size: ", mean_size)
    print("std_deviation: ", std_deviation)
    print("threshold:", threshold)

    # 选择大于阈值的社团
    selected_communities = [community for community in structure_community if len(community) > threshold]
    K = len(selected_communities)
    print("筛选后的{}个社团的节点数量：".format(K), end='')
    for i in selected_communities:
        print(len(i), end=' ')
    print()
    current_comm = selected_communities
    text_datasets = Dataset.from_list([{"text": t} for t in text])
    feat = ori_feat
    model = MambaTextClassification.from_pretrained("state-spaces/mamba-130m", num_class=K)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    for e in range(0, epoch):
        print(f"=================================第{e}次循环=================================")
        print("*****************************GCN社团检测***************************")
        partition, acc = GCN_train(dataset_name,feat, adj, edge, label, graph, current_comm)

        if e < (epoch-1):
            dataset_with_label = Dataset.from_list([
                {"text": t, "label": int(label)} for t, label in zip(text, partition)
            ])
            mamba_dataset = dataset_with_label.train_test_split(test_size=0.1, seed=0)
            mamba_dataset_dict = DatasetDict({
                'train': mamba_dataset['train'],
                'test': mamba_dataset['test']
            })
            print("******************************Mamba模型分类任务训练***************************")
            mamba_train(mamba_dataset_dict, dataset_name, model, tokenizer, e)
            print("******************************训练模型语义特征提取*****************************")
            mamba_feat, pre_labels = semantic_feat_extra(text_datasets, dataset_name)
            # if e>=2:
            #     print(torch.allclose(feat,mamba_feat))
            feat = mamba_feat.to('cpu')
            if e == 0:
                max_acc = acc
            else:
                if e>=10 and e%2 == 0:
                    # HDBSCAN = hdbscan.HDBSCAN(min_cluster_size=10)
                    # Sem_labels = HDBSCAN.fit_predict(feat)
                    # dbscan = DBSCAN(eps=0.1, min_samples=60, metric='cosine')  # 使用余弦距离
                    # DBS_labels = dbscan.fit_predict(feat)
                    # print("Sem labels:",pre_labels)
                    # print("DBSCAN labels number:",max(Sem_labels)+1)
                    # print("DBSCAN labels == -1:",np.sum(Sem_labels == -1))
                    DBSCAN_comm = []
                    # K = max(Sem_labels)+1
                    pre_labels = pre_labels.cpu()
                    # print(pre_labels)
                    unique_labels = set(pre_labels.numpy())
                    # print(unique_labels)
                    for DBS_label in unique_labels:
                        # print("Sem label:",DBS_label)
                        if DBS_label != -1:
                            comm_nodes = set(np.where(pre_labels == DBS_label)[0])
                            # print(comm_nodes)
                            DBSCAN_comm.append(comm_nodes.copy())
                    # print(len(DBSCAN_comm))
                    current_comm = DBSCAN_comm
                    # print(current_comm)
                    # print(len(current_comm))
                    # model.classification_head.num_class = K
                else:
                    # max_acc = acc
                    current_comm = selected_communities

        if e == (epoch-1):
            print(f"{K}-means:")
            Kmeans_model = cluster.KMeans(n_clusters=K, random_state=42)
            cluster_labels = Kmeans_model.fit_predict(feat)
            pre = Kmeans_model.labels_
            km_dbi = davies_bouldin_score(feat, pre)
            km_sc = silhouette_score(feat, pre, metric='cosine')

            km_nmi, km_acc, km_f1, km_ari, km_fmi = result(pre, label)
            km_q = Q.compute_modularity(graph, pre)
            print("km_dbi,km_q,km_nmi,km_acc,km_f1,km_ari,km_sc,km_fmi", km_dbi, km_q, km_nmi, km_acc, km_f1, km_ari, km_sc, km_fmi)
