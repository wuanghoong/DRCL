import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn.inits import reset, uniform
from sklearn.metrics.pairwise import euclidean_distances

from torch_scatter import scatter_mean
from torch.distributions import Categorical

EPS = 1e-15

# def cluster(data, k, temp, num_iter, init = None, cluster_temp=5):
#     '''
#     pytorch (differentiable) implementation of soft k-means clustering.
#     '''
#     #normalize x so it lies on the unit sphere
#     data = torch.diag(1./torch.norm(data, p=2, dim=1)) @ data
#     #use kmeans++ initialization if nothing is provided
#     if init is None:
#         data_np = data.detach().numpy()
#         norm = (data_np**2).sum(axis=1)
#         init = sklearn.cluster.k_means_._k_init(data_np, k, norm, sklearn.utils.check_random_state(None))
#         init = torch.tensor(init, requires_grad=True)
#         if num_iter == 0: return init
#     #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
#     mu = init.to(device)
#     n = data.shape[0]
#     d = data.shape[1]
# #    data = torch.diag(1./torch.norm(data, dim=1, p=2))@data
#     for t in range(num_iter):
#         #get distances between all data points and cluster centers
# #        dist = torch.cosine_similarity(data[:, None].expand(n, k, d).reshape((-1, d)), mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
#         dist = data @ mu.t()
#         #cluster responsibilities via softmax
#         r = torch.softmax(cluster_temp*dist, 1)
#         #total responsibility of each cluster
#         cluster_r = r.sum(dim=0)
#         #mean of points in each cluster weighted by responsibility
#         cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
#         #update cluster means
#         new_mu = torch.diag(1/cluster_r) @ cluster_mean
#         mu = new_mu
#     dist = data @ mu.t()
#     r = torch.softmax(cluster_temp*dist, 1)
#     return mu, r, dist

class DeepGraphInfomax(torch.nn.Module):
    r"""The Deep Graph Infomax model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (Module): The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """

    def __init__(self, hidden_channels, encoder, summary, corruption, args, cluster):
        super(DeepGraphInfomax, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.reset_parameters()
        self.K = args.K
        self.cluster_temp = args.clustertemp
        self.init = torch.rand(self.K,hidden_channels)
        self.cluster = cluster

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)

    def forward(self, *args, **kwargs):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""
        pos_z = self.encoder(*args, **kwargs)#GCN学习节点表示
        # cor = self.corruption(*args, **kwargs) 负样本
        # cor = cor if isinstance(cor, tuple) else (cor, )
        # neg_z = self.encoder(*cor, None)
        summary = self.summary(pos_z)
        num_iter = 1
        pos_z = torch.diag(1. / torch.norm(pos_z, p=2, dim=1)) @ pos_z #节点表示进行L2归一化处理
        center = []
        for comm in args[2]:

            community = pos_z.index_select(0, torch.tensor(list(comm)))
            ct = torch.mean(community, dim=0)
            center.append(ct)
        # print("center:", center)
        mu = torch.stack(center, dim=0)  # 根据Louvain算法得到的中心
        # print("mu:",mu,)
        # mu_stru = pos_z.index_select(0,torch.tensor(args[2]))
        # print("结构中心：",mu_stru,mu_stru.shape)
        dist = pos_z @ mu.t()
        r = torch.softmax(self.cluster_temp * dist, 1)
        r_assign = r.argmax(dim=1)
        r_comm = []
        neg_comm = []
        newu = []
        for label in torch.unique(r_assign):
            cluster_points = pos_z[r_assign == label]
            r_comm.append(cluster_points)
            neg_comm.append((pos_z[r_assign !=label]))
            newu.append(cluster_points.mean(dim=0))
        u = torch.stack(newu, dim=0)  # r分配矩阵预测中心
        # dist_u = torch.cdist(u, u, p=2)  # 欧几里得距离
        # index = dist_u.argmax(dim=1)
        # # print(index)
        # neg_u = u[index]

        return pos_z, mu,  r,  dist, u, r_comm, neg_comm

    def discriminate(self, z, summary, sigmoid=True):
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (Tensor): The latent space.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        #print("shape", z.shape,summary.shape)
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        r"""Computes the mutal information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()

        # print('pos_loss = {}, neg_loss = {}'.format(pos_loss, neg_loss))
        # bin_adj_nodiag = bin_adj * (torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
        # modularity = (1./bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()
        return pos_loss + neg_loss #+ modularity

    def community_dists_probs(self, dist, edge_index, alpha=0.9):
        # dot_products = (self.psi[None, :, :] * z[:, None, :]).sum(dim=2)
        row, col = edge_index
        dot_products_avg_over_Ni = scatter_mean(src=dist[row], index=col, dim=0, dim_size=dist.size(0))
        weighted_dot_products = alpha * dist + (1 - alpha) * dot_products_avg_over_Ni
        return Categorical(logits=dist), Categorical(logits=weighted_dot_products)

    def comm_loss(self,pos_z,mu):
        # print("summary(mu):",self.summary(mu),self.summary(mu).shape)
        # print("torch.log(self.discriminate(pos_z,self.summary(mu),sigmoid=True) + EPS) :",(self.discriminate(pos_z,self.summary(mu),sigmoid=True) + EPS).shape)
        return -torch.log(self.discriminate(pos_z,self.summary(mu), sigmoid=True) + EPS).mean()
    #
    # def modularity(self, mu, r, embeds, dist, bin_adj, mod, args):
    #     bin_adj_nodiag = bin_adj * (torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
    #     # print("bin_adj_nodiag.dtype: ", bin_adj_nodiag.dtype, "\n r.dtype:", r.dtype, "\nmod.dtype:", mod.dtype)
    #     loss = (1. / bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()
    #     return -loss

    def modularity(self, mu, r, embeds, dist, bin_adj, mod, args):
        # bin_adj: dense tensor, mod: dense tensor
        eye = torch.eye(bin_adj.shape[0], device=bin_adj.device)
        bin_adj_nodiag = bin_adj * (1 - eye)  # 去掉对角线
        norm = bin_adj_nodiag.sum()

        if norm.item() == 0:
            return torch.tensor(0.0, device=bin_adj.device)

        # trace(r^T @ B @ r)
        score = torch.trace(r.t() @ mod @ r)
        loss = (1. / norm) * score
        return -loss

    def sd_loss(self, u, r_comm, neg_comm, n):
        L_sd = 0
        k = len(r_comm)
        for i in range(k):
            L_sd = torch.log(self.discriminate(r_comm[i], u[i], sigmoid=True) + EPS).sum() + torch.log(1 - self.discriminate(neg_comm[i], u[i], sigmoid=True) + EPS).sum()
        return -L_sd/(2*n)

    def contrastive_loss1(self, Z, r, neighbor_mask, temperature=0.07):
        # 计算基于社团划分的对比损失函数,同社团节点互为正样本，非同社团节点作为负样本。
        device = Z.device
        # N = Z.shape[0]  # 节点数量
        # 计算相似度矩阵
        sim_matrix = torch.matmul(Z, Z.T) / temperature
        sim_matrix_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_matrix_max.detach()  # 各个节点之间的相似度
        dig = torch.eye(sim_matrix.shape[0], dtype=torch.bool).to(sim_matrix.device)
        sim_matrix.masked_fill_(dig, 0)
        print("所有节点之间的相似度：", sim_matrix)

        # 获取正负样本的掩码
        labels = r.unsqueeze(1)  # (N, 1) 形状
        print(" labels：", labels)
        mask_pos = torch.eq(labels, labels.T).float().to(device)  
        print("正样本掩码矩阵：", mask_pos, mask_pos.shape)
        count_pos = (mask_pos == 1.).sum()
        print("总共的正样本掩码为1的数量：", count_pos)

        mask_neg = 1.0 - mask_pos  # 不同社团为负样本
        print("负样本掩码矩阵：", mask_neg, mask_neg.shape)
        count_neg = (mask_neg == 1.).sum()
        print("总共的正样本掩码为1的数量：", count_neg)

        # 正样本对数分布
        exp_sim_pos = torch.exp(sim_matrix) * mask_pos
        print(" 所有的正样本的指数概率分数：", exp_sim_pos, exp_sim_pos.shape)
        # log_prob_pos = torch.log(exp_sim_pos.sum(1, keepdim=True))
        log_prob_pos = sim_matrix - torch.log(exp_sim_pos.sum(1, keepdim=True))
        print("log_pos:", log_prob_pos, log_prob_pos.shape)
        
        # 负样本对数分布
        exp_sim_neg = torch.exp(sim_matrix) * mask_neg
        print("torch.log(exp_sim_neg.sum(1, keepdim=True))", torch.log(exp_sim_neg.sum(1, keepdim=True)))
        # log_prob_neg = torch.log(exp_sim_neg.sum(1, keepdim=True))
        log_prob_neg = sim_matrix - torch.log(exp_sim_neg.sum(1, keepdim=True))
        print("log_neg", log_prob_neg, log_prob_pos.shape)
        
        # 对比损失
        loss = -log_prob_pos+log_prob_neg
        loss = loss.mean(dim=1)
        loss = loss.mean(dim=0)

        print("loss:", loss)
        return -loss

    def contrastive_loss2(self, Z, r, mask_neighbor, temperature=0.07):
        # 对比损失函数，同社团的一阶/二阶邻居为正样本，最近社团节点为负样本，分母为负样本相加
        device = Z.device
        N = Z.shape[0]  # 节点数量

        # 计算相似度矩阵
        sim_matrix = torch.matmul(Z, Z.T) / temperature
        print("未减最大值的相似度矩阵：", sim_matrix)
        sim_matrix_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_matrix_max.detach()
        dig = torch.eye(sim_matrix.shape[0], dtype=torch.bool).to(sim_matrix.device)
        sim_matrix.masked_fill_(dig, 0)
        print("所有节点之间的相似度：", sim_matrix)

        # 相同社团一/二阶邻居正负样本的掩码
        labels = r.unsqueeze(1)  # (N, 1) 形状
        print(" labels：", labels)
        mask_neighbor = torch.from_numpy(mask_neighbor)
        mask_comm = torch.eq(labels, labels.T).float()  # 与该节点同社团的节点掩码为1
        mask_neighbor = mask_neighbor.to(torch.float)
        mask_comm = mask_comm.to(torch.float)
        mask_pos = (mask_neighbor.bool() & mask_comm.bool()).float()   # 社团掩码与邻居掩码的逻辑与（表示同一社团的邻居节点为1）作为正样本掩码
        print("正样本掩码矩阵：", mask_pos, mask_pos.shape)
        count_pos = (mask_pos == 1.).sum()
        print("总共的正样本掩码为1的数量：", count_pos)

        # 社团均值中心计算
        num_communities = r.max() + 1  # 假设标签是连续的0~num_communities-1
        dim = Z.size(1)  # 节点特征的维度
        community_sums = torch.zeros(num_communities, dim)  # 保存一个社团的所有节点特征和
        community_counts = torch.zeros(num_communities, dtype=torch.float32)  # 储存每个社团的节点数量
        community_sums.scatter_add_(0, r.unsqueeze(1).expand(-1, dim), Z)  # 索引标签，将相同标签的特征进行累加到对应的索引上
        community_counts.scatter_add_(0, r, torch.ones_like(r, dtype=torch.float32))  # 将相同社团的数量累加到相对应的索引上
        community_centers = community_sums / community_counts.unsqueeze(1)  # 计算均值得到特征中心矩阵
        print("community_centers", community_centers, community_centers.shape)

        # 最近社团负样本掩码
        squared_sum = torch.sum(community_centers ** 2, dim=1, keepdim=True)  # 计算均值中心的平方和，方便后续计算欧氏距离
        distance_matrix = torch.sqrt(
            torch.clamp(squared_sum - 2 * torch.matmul(community_centers, community_centers.T) + squared_sum.T, min=0)
        )
        unique_labels = r.unique()
        large_value = 1e12  # 添加大值，防止自环，出现自身到自身的距离错误
        distance_matrix = distance_matrix + torch.eye(distance_matrix.size(0),
                                                      device=distance_matrix.device) * large_value
        closest_indices = torch.argmin(distance_matrix, dim=1)
        closest_communities = unique_labels[closest_indices]
        closest_communities_for_each = closest_communities[r].unsqueeze(1)  # (num_nodes, 1)
        mask_neg = (r.unsqueeze(0) == closest_communities_for_each).float()
        print("负样本掩码矩阵：", mask_neg, mask_neg.shape)
        count_neg = (mask_neg == 1.).sum()
        print("总共的负样本掩码为1的数量：", count_neg)


        # 正样本对数分布
        eps = 1e-10
        exp_sim_pos = torch.exp(sim_matrix) * mask_pos
        has_pos_samples = (mask_pos.sum(1) > 0).bool()
        log_prob_pos = sim_matrix - torch.log(exp_sim_pos.sum(1, keepdim=True) + eps)
        default_loss_value = torch.tensor(1e-10).to(sim_matrix.device)  # 可根据需要调整
        log_prob_pos = torch.where(has_pos_samples.unsqueeze(1), log_prob_pos, default_loss_value)
        print(" 所有的正样本的指数概率分数：", exp_sim_pos, exp_sim_pos.shape)
        # # log_prob_pos = torch.log(exp_sim_pos.sum(1, keepdim=True))
        # log_prob_pos = sim_matrix - torch.log(exp_sim_pos.sum(1, keepdim=True))
        print("log_pos:", log_prob_pos, log_prob_pos.shape)

        # 负样本对数分布
        exp_sim_neg = torch.exp(sim_matrix) * mask_neg
        # print("torch.log(exp_sim_neg.sum(1, keepdim=True))", torch.log(exp_sim_neg.sum(1, keepdim=True)))
        # log_prob_neg = torch.log(exp_sim_neg.sum(1, keepdim=True))
        log_prob_neg = sim_matrix - torch.log(exp_sim_neg.sum(1, keepdim=True))
        print("log_neg", log_prob_neg, log_prob_pos.shape)

        loss = -log_prob_pos + log_prob_neg
        loss = loss.mean(dim=1)
        print("各节点loss：", loss)
        loss = -loss.mean(dim=0)

        # loss = -log_prob_pos.mean() + log_prob_neg.mean()
        # loss = loss/N

        # loss = -log_prob_pos + log_prob_neg
        # loss = loss.mean()
        print("loss:", loss)
        print("mask_pos sum: ", mask_pos.sum(dim=1))  # 检查某个节点的是否存在正样本（即是否存在邻居处在一个社团）
        print("mask_neg sum: ", mask_neg.sum(dim=1))  #
        return loss

    def contrastive_loss3(self, Z, r, mask_neighbor, temperature=0.07):
        # 对比损失函数，同社团的一阶/二阶邻居+社团中心为正样本，最近社团节点为负样本，分母为负样本相加
        device = Z.device
        N = Z.shape[0]  # 节点数量

        # 计算相似度矩阵
        sim_matrix = torch.matmul(Z, Z.T) / temperature
        print("未减最大值的相似度矩阵：", sim_matrix)
        sim_matrix_max, _ = sim_matrix.max(dim=1, keepdim=True)
        print(sim_matrix)
        sim_matrix = sim_matrix - sim_matrix_max.detach()
        dig = torch.eye(sim_matrix.shape[0], dtype=torch.bool).to(sim_matrix.device)
        sim_matrix.masked_fill_(dig, 0)
        print("所有节点之间的相似度：", sim_matrix)

        # 相同社团一/二阶邻居正负样本的掩码
        labels = r.unsqueeze(1)  # (N, 1) 形状
        print(" labels：", labels)
        mask_neighbor = torch.from_numpy(mask_neighbor)
        mask_comm = torch.eq(labels, labels.T).float()  # 与该节点同社团的节点掩码为1
        mask_neighbor = mask_neighbor.to(torch.float)
        mask_comm = mask_comm.to(torch.float)
        mask_pos = (mask_neighbor.bool() & mask_comm.bool()).float()   # 社团掩码与邻居掩码的逻辑与（表示同一社团的邻居节点为1）
        identity_matrix = torch.eye(N, dtype=torch.float)
        mask_pos = mask_pos + identity_matrix  # 用对角线表示中心，进行掩码为1
        print("正样本掩码矩阵：", mask_pos, mask_pos.shape)
        count_pos = (mask_pos == 1.).sum()
        print("总共的正样本掩码为1的数量：", count_pos)

        # 社团均值中心计算
        num_communities = r.max() + 1  # 假设标签是连续的0~num_communities-1
        dim = Z.size(1)  # 节点特征的维度
        community_sums = torch.zeros(num_communities, dim)  # 保存一个社团的所有节点特征和
        community_counts = torch.zeros(num_communities, dtype=torch.float32)  # 储存每个社团的节点数量
        community_sums.scatter_add_(0, r.unsqueeze(1).expand(-1, dim), Z)  # 索引标签，将相同标签的特征进行累加到对应的索引上
        community_counts.scatter_add_(0, r, torch.ones_like(r, dtype=torch.float32))  # 将相同社团的数量累加到相对应的索引上
        community_centers = community_sums / community_counts.unsqueeze(1)  # 计算均值得到特征中心矩阵
        print("community_centers", community_centers, community_centers.shape)

        # 每个节点和其社团中心的相似度计算
        community_centers_for_each = community_centers[r]  # (N, d)，根据节点的社团分配选择均值中心
        print("community_centers_for_each", community_centers_for_each, community_centers_for_each.shape)
        sim_with_center = torch.matmul(Z, community_centers_for_each.T) / temperature
        print("sim_with_center", sim_with_center, sim_with_center.shape)
        sim_with_center = sim_with_center - sim_matrix_max.detach()  # 注意形状为[n,n]，只有对角线才是相同社团中心的相似度
        sim_center_dig = torch.zeros_like(sim_with_center)
        sim_center_dig += torch.diag(sim_with_center.diagonal())
        print("sim_center_dig", sim_center_dig, sim_center_dig.shape)
        sim_matrix = sim_matrix + sim_center_dig  # 将空的对角线，填入中心相似度

        # 最近社团负样本掩码
        squared_sum = torch.sum(community_centers ** 2, dim=1, keepdim=True)  # 计算均值中心的平方和，方便后续计算欧氏距离
        distance_matrix = torch.sqrt(
            torch.clamp(squared_sum - 2 * torch.matmul(community_centers, community_centers.T) + squared_sum.T, min=0)
        )
        unique_labels = r.unique()
        large_value = 1e12  # 添加大值，防止自环，出现自身到自身的距离错误
        distance_matrix = distance_matrix + torch.eye(distance_matrix.size(0),
                                                      device=distance_matrix.device) * large_value
        closest_indices = torch.argmin(distance_matrix, dim=1)
        closest_communities = unique_labels[closest_indices]
        closest_communities_for_each = closest_communities[r].unsqueeze(1)  # (num_nodes, 1)
        mask_neg = (r.unsqueeze(0) == closest_communities_for_each).float()
        print("负样本掩码矩阵：", mask_neg, mask_neg.shape)
        count_neg = (mask_neg == 1.).sum()
        print("总共的负样本掩码为1的数量：", count_neg)


        # 正样本对数分布

        exp_sim_pos = torch.exp(sim_matrix) * mask_pos
        print(" 所有的正样本的指数概率分数：", exp_sim_pos, exp_sim_pos.shape)
        # log_prob_pos = torch.log(exp_sim_pos.sum(1, keepdim=True))
        log_prob_pos = sim_matrix - torch.log(exp_sim_pos.sum(1, keepdim=True))
        print("log_pos:", log_prob_pos, log_prob_pos.shape)

        # 负样本对数分布
        exp_sim_neg = torch.exp(sim_matrix) * mask_neg
        # print("torch.log(exp_sim_neg.sum(1, keepdim=True))", torch.log(exp_sim_neg.sum(1, keepdim=True)))
        # log_prob_neg = torch.log(exp_sim_neg.sum(1, keepdim=True))
        log_prob_neg = sim_matrix - torch.log(exp_sim_neg.sum(1, keepdim=True))
        print("log_neg", log_prob_neg, log_prob_pos.shape)

        loss = -log_prob_pos + log_prob_neg
        loss = loss.mean(dim=1)
        print("各节点loss：", loss)
        loss = -loss.mean(dim=0)

        # loss = -log_prob_pos.mean() + log_prob_neg.mean()
        # loss = loss/N

        # loss = -log_prob_pos + log_prob_neg
        # loss = loss.mean()
        print("loss:", loss)
        print("mask_pos sum: ", mask_pos.sum(dim=1))  # 检查某个节点的是否存在正样本（即是否存在邻居处在一个社团）
        print("mask_neg sum: ", mask_neg.sum(dim=1))  #
        return loss

    def contrastive_loss4(self, Z, r, mask_neighbor, temperature=0.07):
        # 对比损失函数，同社团的一阶/二阶邻居为正样本，最近社团节点为负样本，分母为正负样本相加
        device = Z.device
        N = Z.shape[0]  # 节点数量

        # 计算相似度矩阵
        sim_matrix = torch.matmul(Z, Z.T) / temperature
        # print("未减最大值的相似度矩阵：", sim_matrix)
        sim_matrix_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_matrix_max.detach()
        dig = torch.eye(sim_matrix.shape[0], dtype=torch.bool).to(sim_matrix.device)
        sim_matrix.masked_fill_(dig, 0)
        # print("所有节点之间的相似度：", sim_matrix)

        # 相同社团一/二阶邻居正负样本的掩码
        labels = r.unsqueeze(1)  # (N, 1) 形状
        # print(" labels：", labels)
        mask_neighbor = torch.from_numpy(mask_neighbor)
        mask_comm = torch.eq(labels, labels.T).float()  # 与该节点同社团的节点掩码为1
        mask_neighbor = mask_neighbor.to(torch.float)
        mask_comm = mask_comm.to(torch.float)
        # mask_pos = mask_neighbor
        mask_pos = (mask_neighbor.bool() & mask_comm.bool()).float()   # 社团掩码与邻居掩码的逻辑与（表示同一社团的邻居节点为1）作为正样本掩码
        # mask_pos = (mask_neighbor.bool() | mask_comm.bool()).float()   # 社团掩码与邻居掩码的逻辑或（表示社团节点与邻居）作为正样本掩码
        # print("正样本掩码矩阵：", mask_pos, mask_pos.shape)
        count_pos = (mask_pos == 1.).sum()
        # print("总共的正样本掩码为1的数量：", count_pos)

        # 社团均值中心计算
        num_communities = r.max() + 1  # 假设标签是连续的0~num_communities-1
        dim = Z.size(1)  # 节点特征的维度
        community_sums = torch.zeros(num_communities, dim)  # 保存一个社团的所有节点特征和
        community_counts = torch.zeros(num_communities, dtype=torch.float32)  # 储存每个社团的节点数量
        community_sums.scatter_add_(0, r.unsqueeze(1).expand(-1, dim), Z)  # 索引标签，将相同标签的特征进行累加到对应的索引上
        community_counts.scatter_add_(0, r, torch.ones_like(r, dtype=torch.float32))  # 将相同社团的数量累加到相对应的索引上
        community_centers = community_sums / community_counts.unsqueeze(1)  # 计算均值得到特征中心矩阵
        # print("community_centers", community_centers, community_centers.shape)

        # 最近社团负样本掩码
        squared_sum = torch.sum(community_centers ** 2, dim=1, keepdim=True)  # 计算均值中心的平方和，方便后续计算欧氏距离
        distance_matrix = torch.sqrt(
            torch.clamp(squared_sum - 2 * torch.matmul(community_centers, community_centers.T) + squared_sum.T, min=0)
        )
        unique_labels = r.unique()
        large_value = 1e12  # 添加大值，防止自环，出现自身到自身的距离错误
        distance_matrix = distance_matrix + torch.eye(distance_matrix.size(0),
                                                      device=distance_matrix.device) * large_value
        closest_indices = torch.argmin(distance_matrix, dim=1)
        closest_communities = unique_labels[closest_indices]
        closest_communities_for_each = closest_communities[r].unsqueeze(1)  # (num_nodes, 1)
        mask_neg = (r.unsqueeze(0) == closest_communities_for_each).float()
        # print("负样本掩码矩阵：", mask_neg, mask_neg.shape)
        count_neg = (mask_neg == 1.).sum()
        # print("总共的负样本掩码为1的数量：", count_neg)

        # 正负样本掩码并集，做分母
        mask = (mask_pos.bool() | mask_neg.bool()).float()   # 正负样本合并

        # 所有元素与正负样本和相除
        exp_sim_sample = torch.exp(sim_matrix) * mask  # 只有正负样本的相似矩阵
        log_all_sample = sim_matrix - torch.log(exp_sim_sample.sum(1, keepdim=True))

        # 正样本与正负样本和相除
        log_pos_sample = log_all_sample * mask_pos

        loss = log_pos_sample.sum(dim=1)  # 计算节点i与节点的正样本之间的损失
        loss = -loss.mean(dim=0)  # 求均值
        print("loss:", loss)

        return loss

    def contrastive_loss42(self, Z, r, mask_Lou, temperature=0.07):
        # 对比损失函数，同社团的一阶/二阶邻居为正样本，最近社团节点为负样本，分母为正负样本相加
        device = Z.device
        N = Z.shape[0]  # 节点数量

        # 计算相似度矩阵
        sim_matrix = torch.matmul(Z, Z.T) / temperature
        # print("未减最大值的相似度矩阵：", sim_matrix)
        sim_matrix_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_matrix_max.detach()
        dig = torch.eye(sim_matrix.shape[0], dtype=torch.bool).to(sim_matrix.device)
        sim_matrix.masked_fill_(dig, 0)
        # print("所有节点之间的相似度：", sim_matrix)

        # 相同社团一/二阶邻居正负样本的掩码
        labels = r.unsqueeze(1)  # (N, 1) 形状
        # print(" labels：", labels)
        mask_Lou = torch.from_numpy(mask_Lou)
        mask_comm = torch.eq(labels, labels.T).float()  # 与该节点同社团的节点掩码为1
        mask_Lou = mask_Lou.to(torch.float)
        mask_comm = mask_comm.to(torch.float)
        # mask_pos = mask_neighbor
        mask_pos = (mask_Lou.bool() & mask_comm.bool()).float()   # 社团掩码与邻居掩码的逻辑与（表示同一社团的邻居节点为1）作为正样本掩码
        # mask_pos = (mask_Lou.bool() | mask_comm.bool()).float()   # 社团掩码与邻居掩码的逻辑或（表示社团节点与邻居）作为正样本掩码
        # print("正样本掩码矩阵：", mask_pos, mask_pos.shape)
        count_pos = (mask_pos == 1.).sum()
        # print("总共的正样本掩码为1的数量：", count_pos)

        # 社团均值中心计算
        num_communities = r.max() + 1  # 假设标签是连续的0~num_communities-1
        dim = Z.size(1)  # 节点特征的维度
        community_sums = torch.zeros(num_communities, dim)  # 保存一个社团的所有节点特征和
        community_counts = torch.zeros(num_communities, dtype=torch.float32)  # 储存每个社团的节点数量
        community_sums.scatter_add_(0, r.unsqueeze(1).expand(-1, dim), Z)  # 索引标签，将相同标签的特征进行累加到对应的索引上
        community_counts.scatter_add_(0, r, torch.ones_like(r, dtype=torch.float32))  # 将相同社团的数量累加到相对应的索引上
        community_centers = community_sums / community_counts.unsqueeze(1)  # 计算均值得到特征中心矩阵
        # print("community_centers", community_centers, community_centers.shape)

        # 最近社团负样本掩码
        squared_sum = torch.sum(community_centers ** 2, dim=1, keepdim=True)  # 计算均值中心的平方和，方便后续计算欧氏距离
        distance_matrix = torch.sqrt(
            torch.clamp(squared_sum - 2 * torch.matmul(community_centers, community_centers.T) + squared_sum.T, min=0)
        )
        unique_labels = r.unique()
        large_value = 1e12  # 添加大值，防止自环，出现自身到自身的距离错误
        distance_matrix = distance_matrix + torch.eye(distance_matrix.size(0),
                                                      device=distance_matrix.device) * large_value
        closest_indices = torch.argmin(distance_matrix, dim=1)
        closest_communities = unique_labels[closest_indices]
        closest_communities_for_each = closest_communities[r].unsqueeze(1)  # (num_nodes, 1)
        mask_neg = (r.unsqueeze(0) == closest_communities_for_each).float()
        # print("负样本掩码矩阵：", mask_neg, mask_neg.shape)
        count_neg = (mask_neg == 1.).sum()
        # print("总共的负样本掩码为1的数量：", count_neg)

        # 正负样本掩码并集，做分母
        mask = (mask_pos.bool() | mask_neg.bool()).float()   # 正负样本合并

        # 所有元素与正负样本和相除
        exp_sim_sample = torch.exp(sim_matrix) * mask  # 只有正负样本的相似矩阵
        log_all_sample = sim_matrix - torch.log(exp_sim_sample.sum(1, keepdim=True))

        # 正样本与正负样本和相除
        log_pos_sample = log_all_sample * mask_pos

        loss = log_pos_sample.sum(dim=1)  # 计算节点i与节点的正样本之间的损失
        loss = -loss.mean(dim=0)  # 求均值
        print("loss:", loss)

        return loss

    def contrastive_loss5(self, Z, r, mask_neighbor, temperature=0.07):
        # 对比损失函数，同社团的一阶/二阶邻居+社团中心为正样本，最近社团节点为负样本，分母为负样本相加
        device = Z.device
        N = Z.shape[0]  # 节点数量

        # 计算相似度矩阵
        sim_matrix = torch.matmul(Z, Z.T) / temperature
        print("未减最大值的相似度矩阵：", sim_matrix)
        sim_matrix_max, _ = sim_matrix.max(dim=1, keepdim=True)
        print(sim_matrix)
        sim_matrix = sim_matrix - sim_matrix_max.detach()
        dig = torch.eye(sim_matrix.shape[0], dtype=torch.bool).to(sim_matrix.device)
        sim_matrix.masked_fill_(dig, 0)
        print("所有节点之间的相似度：", sim_matrix)

        # 相同社团一/二阶邻居正负样本的掩码
        labels = r.unsqueeze(1)  # (N, 1) 形状
        print(" labels：", labels)
        mask_neighbor = torch.from_numpy(mask_neighbor)
        mask_comm = torch.eq(labels, labels.T).float()  # 与该节点同社团的节点掩码为1
        mask_neighbor = mask_neighbor.to(torch.float)
        mask_comm = mask_comm.to(torch.float)
        mask_pos = (mask_neighbor.bool() & mask_comm.bool()).float()   # 社团掩码与邻居掩码的逻辑与（表示同一社团的邻居节点为1）
        identity_matrix = torch.eye(N, dtype=torch.float)
        mask_pos = mask_pos + identity_matrix  # 用对角线表示中心，进行掩码为1
        print("正样本掩码矩阵：", mask_pos, mask_pos.shape)
        count_pos = (mask_pos == 1.).sum()
        print("总共的正样本掩码为1的数量：", count_pos)

        # 社团均值中心计算
        num_communities = r.max() + 1  # 假设标签是连续的0~num_communities-1
        dim = Z.size(1)  # 节点特征的维度
        community_sums = torch.zeros(num_communities, dim)  # 保存一个社团的所有节点特征和
        community_counts = torch.zeros(num_communities, dtype=torch.float32)  # 储存每个社团的节点数量
        community_sums.scatter_add_(0, r.unsqueeze(1).expand(-1, dim), Z)  # 索引标签，将相同标签的特征进行累加到对应的索引上
        community_counts.scatter_add_(0, r, torch.ones_like(r, dtype=torch.float32))  # 将相同社团的数量累加到相对应的索引上
        community_centers = community_sums / community_counts.unsqueeze(1)  # 计算均值得到特征中心矩阵
        print("community_centers", community_centers, community_centers.shape)

        # 每个节点和其社团中心的相似度计算
        community_centers_for_each = community_centers[r]  # (N, d)，根据节点的社团分配选择均值中心
        print("community_centers_for_each", community_centers_for_each, community_centers_for_each.shape)
        sim_with_center = torch.matmul(Z, community_centers_for_each.T) / temperature
        print("sim_with_center", sim_with_center, sim_with_center.shape)
        sim_with_center = sim_with_center - sim_matrix_max.detach()  # 注意形状为[n,n]，只有对角线才是相同社团中心的相似度
        sim_center_dig = torch.zeros_like(sim_with_center)
        sim_center_dig += torch.diag(sim_with_center.diagonal())
        print("sim_center_dig", sim_center_dig, sim_center_dig.shape)
        # 节点与中心的相似度加入相似度矩阵
        sim_matrix = sim_matrix + sim_center_dig  # 将空的对角线，填入中心相似度

        # 最近社团负样本掩码
        squared_sum = torch.sum(community_centers ** 2, dim=1, keepdim=True)  # 计算均值中心的平方和，方便后续计算欧氏距离
        distance_matrix = torch.sqrt(
            torch.clamp(squared_sum - 2 * torch.matmul(community_centers, community_centers.T) + squared_sum.T, min=0)
        )
        unique_labels = r.unique()
        large_value = 1e12  # 添加大值，防止自环，出现自身到自身的距离错误
        distance_matrix = distance_matrix + torch.eye(distance_matrix.size(0),
                                                      device=distance_matrix.device) * large_value
        closest_indices = torch.argmin(distance_matrix, dim=1)
        closest_communities = unique_labels[closest_indices]
        closest_communities_for_each = closest_communities[r].unsqueeze(1)  # (num_nodes, 1)
        mask_neg = (r.unsqueeze(0) == closest_communities_for_each).float()
        print("负样本掩码矩阵：", mask_neg, mask_neg.shape)
        count_neg = (mask_neg == 1.).sum()
        print("总共的负样本掩码为1的数量：", count_neg)

        # 正负样本掩码并集，做分母
        mask = (mask_pos.bool() | mask_neg.bool()).float()  # 正负样本合并

        # 所有元素与正负样本和相除
        exp_sim_sample = torch.exp(sim_matrix) * mask  # 只有正负样本的相似矩阵
        log_all_sample = sim_matrix - torch.log(exp_sim_sample.sum(1, keepdim=True))

        # 正样本与正负样本和相除
        log_pos_sample = log_all_sample * mask_pos

        loss = log_pos_sample.sum(dim=1)  # 计算节点i与节点的正样本之间的损失
        loss = -loss.mean(dim=0)  # 求均值
        print("loss:", loss)

        return loss


    def test(self, train_z, train_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)
