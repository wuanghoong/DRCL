from model import Encoder, corruption, Summarizer, cluster_net
from utils.load_data import load_data
from DGI import DeepGraphInfomax
import Q
import evaluation

from datetime import datetime
from sklearn.metrics import davies_bouldin_score, silhouette_score,fowlkes_mallows_score
# from clusteval import clusteval
from sklearn import cluster

from datetime import datetime, timezone, timedelta
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import argparse
import torch
import time
import os

from openpyxl import load_workbook
from openpyxl import Workbook

from torch.distributions import kl_divergence#KL散度
from sklearn.metrics import f1_score
import scipy.sparse as sp



parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate.')  # 在优化算法中，控制迭代更新参数的步长
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')  # 隐藏层大小
parser.add_argument('--datasets', type=str, default="cora",  # cora, citeseer, acm, amap, film, pubmed,
                    help='which network to load')   # cora 9 , citeseer 11, acm 10, amap 7, film 2, pubmed 4,
parser.add_argument('--color', type=str, default='r-',
                    help='color line')
parser.add_argument('--K', type=int, default=7,
                    help='How many partitions')
parser.add_argument('--clustertemp', type=float, default=30,
                    help='how hard to make the softmax for the cluster assignments')
parser.add_argument('--train_iters', type=int, default=1001,
                    help='number of training iterations')
parser.add_argument('--num_cluster_iter', type=int, default=1,
                    help='number of iterations for clustering')
parser.add_argument('--seed', type=int, default=24, help='Random seed.')  # 随机种子，确保随机过程的可复现性


args = parser.parse_args()


def make_modularity_matrix(adj):
    adj = adj*(torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - degrees@degrees.t()/adj.sum()
    return mod
def make_modularity_sp_matrix(adj):
    degrees = np.array(adj.sum(axis=1)).flatten()
    m = adj.sum()
    B = adj - np.outer(degrees, degrees) / m
    return sp.csr_matrix(B)


def count(label):
    cnt = [0] * 40
    for i in label:
        cnt[i] += 1
    print(cnt)
    return cnt


def result(pred, labels):
    # print("3:",pred)
    nmi = evaluation.NMI_helper(pred, labels)
    # print("4:",pred)
    f1 = evaluation.cal_F_score(pred, labels)[0]
    # f1 = f1_score(labels, pred, average='micro')
    # print("6:",pred)
    ari = evaluation.adjusted_rand_score(pred, labels)
    # print("7:",pred)
    ac = evaluation.matched_ac(pred, labels)
    # print("5:",pred)
    fmi = fowlkes_mallows_score(labels,pred)
    return nmi, ac, f1, ari, fmi


def train(model,optimizer,feat,edge,selected_communities,adj,test_object):
    model.train()
    optimizer.zero_grad()
    pos_z, mu, r, dist, u, r_comm, neg_comm = model(feat, edge, selected_communities)

    modularity_loss = model.modularity(mu, r, pos_z, dist, adj, test_object, args)

    loss = 1.0 * modularity_loss  # + c * kl_loss
    print("Loss:", loss)
    # loss = dgi_loss
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model,graph,feat, edge, selected_communities,label):
    model.eval()

    with torch.no_grad():
        node_emb, mu, r, _, _, _, _ = model(feat, edge, selected_communities)
    print(r.shape)
    r_assign = r.argmax(dim=1)
    print(max(r_assign))
    # print("1:",r_assign)
    print('label is:')
    _ = count(label)
    print('result of r_assign is:')
    _ = count(r_assign)
    # print("2:",r_assign)
    # nmi,ac,f1,ari = result(pred, labels)
    r_nmi, r_ac, r_f1, r_ari, r_fmi = result(r_assign.numpy(), label)  # numpy
    # print("1")
    r_assign = r.argmax(dim=1)

    # print("8:",r_assign)
    if len(np.unique(r_assign)) > 1:
        DBI = davies_bouldin_score(node_emb, r_assign)  # tensor
        SC = silhouette_score(node_emb,r_assign,metric='cosine')
        DI = 0
        DI2 = 0
        q = Q.compute_modularity(graph, r_assign)
        DBIof = 0
        DIof = 0
    else:
        SC=0
        DBI = 3
        DI = -3
        DI2 = -3
        q = 0
        DBIof = 3
        DIof = -3
        # rs_nmi,rs_ac,rs_f1,rs_ari = result(r_s_assign.numpy(),labels)

    # print("Kmeans Metrics: ",nmi,ac,f1,ari)
    print("New Center Metrics: ", DBI, q, r_nmi, r_ac, r_f1, r_ari, SC, r_fmi)
    # print("Dunn Index:",dunn_index)
    # print("9:",r_assign)
    return node_emb, r_assign, r_nmi, r_ac, r_f1, r_ari, DBI, DI, DI2, q, DBIof, DIof, SC, r_fmi


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
alpha = [0.0]
beta = [0.001]  # [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1, -1.5, -2]
# gamma = [0.0, 0.1]
gamma = [0.0]  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2]

# for args.datasets in ["cora"]:  # "cora", "citeseer", "acm", "amap", "film", "pubmed", "cocs", "amac", "uat"
start_time = time.perf_counter()

# print("****************************", args.datasets, "datasets ******************************")

def GCN_train(dataset_name,feat,adj,edge,label,graph,selected_communities):
    if dataset_name == 'photo':
        test_object = make_modularity_matrix(adj)
    elif dataset_name == 'cora' or 'citeseer' or 'wikics' or 'arxiv_2023' or 'pubmed':
        mod_sp = make_modularity_sp_matrix(adj)
        test_object = torch.from_numpy(mod_sp.toarray()).float().to(device)
        adj = torch.from_numpy(adj.toarray()).float().to(device)
    num_features = feat.shape[1]
    print("label:", label)

    K = len(selected_communities)
    print(K)

    args.K = K
    args.cuda = torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setting up the model and optimizer
    hidden_size = args.hidden
    model = DeepGraphInfomax(
        hidden_channels=hidden_size, encoder=Encoder(num_features, hidden_size),
        summary=Summarizer(),
        corruption=corruption,
        args=args,
        cluster=cluster_net).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)

    # comm_loss，modularity_loss,sd_loss 的系数 a,b,c

    max_nmi = 0
    max_ac = 0
    max_ari = 0
    max_f1 = 0
    min_dbi = 3
    max_di = -3
    max_di2 = -3
    max_q = 0
    min_dbiof = 20
    max_diof = -3
    max_sc = 0
    max_fmi = 0

    dbi_values = []
    di_values = []
    di2_values = []
    q_values = []

    stop_cnt = 0
    best_idx = 0
    patience = 200
    min_loss = 1e9
    real_epoch = 0
    # print(feat.device,edge.device,selected_communities.deviece,adj.device,test_object.device)
    for epoch in range(1, 301):
        loss = train(model,optimizer,feat,edge,selected_communities,adj,test_object)
        if epoch % 2 == 0 and epoch > 0:
            print('epoch = {}'.format(epoch))
            final_z, final_r, tmp_max_nmi, tmp_max_ac, tmp_max_f1, tmp_max_ari, tmp_max_dbi, tmp_max_di, tmp_max_di2, tmp_max_q, tmp_max_dbiof, tmp_max_diof, tmp_max_sc, tmp_max_fmi= test(model,graph,feat, edge, selected_communities,label)

            dbi_values.append(tmp_max_dbi)
            di_values.append(tmp_max_di)
            di2_values.append(tmp_max_di2)
            q_values.append(tmp_max_q)
            if tmp_max_ac>max_ac:
                best_idx = epoch
                torch.save(model.state_dict(),'best_model.pkl')
            max_nmi = max(max_nmi, tmp_max_nmi)
            max_ac = max(max_ac, tmp_max_ac)
            max_f1 = max(max_f1, tmp_max_f1)
            max_ari = max(max_ari, tmp_max_ari)
            min_dbi = min(min_dbi, tmp_max_dbi)
            max_di = max(max_di, tmp_max_di)
            max_di2 = max(max_di2, tmp_max_di2)
            max_q = max(max_q, tmp_max_q)
            min_dbiof = min(min_dbiof, tmp_max_dbiof)
            max_diof = max(max_diof, tmp_max_diof)
            max_sc = max(max_sc, tmp_max_sc)
            max_fmi = max(max_fmi, tmp_max_fmi)
            # node_classification_test(model)
            print("----------------------------------------------------------")
        if loss < min_loss:
            min_loss = loss
            # best_idx = epoch
            stop_cnt = 0
            # torch.save(model.state_dict(), 'best_model.pkl')

        else:
            stop_cnt += 1
        if stop_cnt >= patience:
            real_epoch = epoch
            break

    print('Loading {}th epoch'.format(best_idx))
    model.load_state_dict(torch.load('best_model.pkl'))
    print('Start testing !!!')
    _, r_assign, _, _, _, _, _, _, _, _, _, _, _, _ = test(model,graph,feat, edge, selected_communities,label)
    # print("5:",r_assign)
    # node_classification_test(model)
    print("min dbi为", min_dbi)
    print("max q:", max_q)
    print("max nmi为", max_nmi)
    print("max ac为:", max_ac)
    print("max f1为:", max_f1)
    print("max ari为:", max_ari)

    # print("max di为", max_di)
    # print("max di2:", max_di2)

    print("max sc:", max_sc)
    print("max fmi:", max_fmi)
    # print("min bgi_of为", min_dbiof)
    # print("max di_of为", max_diof)
    now_time = datetime.now()
    end_time = time.perf_counter()
    running_time = end_time - start_time
    print(f"The running time:{running_time}s")
    cnt = count(r_assign)

    file = open("result.csv", "a+")
    print(f"{now_time}-----------------", file=file)
    print("\tmin DBI:\t", min_dbi,
          # "\n\tmax DI:\t", max_di2,
          "\n\tmax Q:\t", max_q,
          "\n\tmax NMI:\t", max_nmi,
          "\n\tmax ACC:\t", max_ac,
          "\n\tmax F1:\t", max_f1,
          "\n\tmax ARI:\t", max_ari,
          "\n\tmax SC:\t", max_sc,
          "\n\tmax FMI:\t", max_fmi,
          "\n\tresult assign:\t",cnt,
          # "\n\tmin DBI_of:\t", min_dbiof,
          # "\n\tmax DI_of:\t", max_diof,
          file=file)
    file.close()
    # print("min label:",min(r_assign),"max label:",max(r_assign))
    # print("6:",r_assign)
    return r_assign, max_ac

