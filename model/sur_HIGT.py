import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch import nn, Tensor

from torch import nn
import torch
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention,GCNConv, ChebConv, SAGEConv, GraphConv, LEConv, LayerNorm, GATConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MIL(nn.Module):
    def __init__(
        self,args,
        # GCN
        gcn_in_channels: int=1024,
        gcn_hid_channels: int=1024,
        gcn_out_channels: int=1024,
        gcn_drop_ratio: float=0.3,
        patch_ratio: int=4,
        pool_ratio: list=[0.5,5],

        # mhit
        re_patch_size: int=64,
        out_classes: int=30,
        mhit_num: int=3,

        # fusion
        fusion_exp_ratio: int=4,

        ) -> None:
        super().__init__()

        self.out_classes = out_classes
        self.ori_patch_size = gcn_out_channels
        self.re_patch_size = re_patch_size

        self.gcn = H2GCN(
            in_feats=gcn_in_channels, 
            n_hidden= gcn_hid_channels, 
            out_feats=gcn_out_channels, 
            drop_out_ratio=gcn_drop_ratio, 
            pool_ratio=pool_ratio,
        )

        global_rep = []

        self.last_pool_ratio = pool_ratio[-1]

        self.patch_ratio = patch_ratio

        for _ in range(mhit_num):
            global_rep.append(
                MobileHIT_Block(
                    channel = 1,
                    re_patch_size = re_patch_size,
                    ori_patch_size = gcn_out_channels,
                    region_node_num = self.last_pool_ratio,
                    patch_node_num = self.patch_ratio*self.last_pool_ratio
                )
            )

        self.mhit = nn.Sequential(*global_rep)

        fusion_in_channel = int(pool_ratio[-1]*patch_ratio*2)
        fusion_out_channel = fusion_in_channel*fusion_exp_ratio
        self.fusion = Fusion_Block(
            in_channel=fusion_in_channel,
            out_channel=fusion_out_channel
        )

        self.ln = nn.LayerNorm(gcn_out_channels)

        self.classifier = nn.Sequential(
            nn.Linear(gcn_out_channels, self.out_classes)
        )

        # init params
        self.apply(self.init_parameters)

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass
    

    def forward(self, data):

        # HI_GCN
        x, edge_index, node_type, tree = self.gcn(data)

        # HI_ViT
        tumbnail_list = torch.where(node_type==0)[0].tolist()
        region_list = torch.where(node_type==1)[0].tolist()
        patch_list = torch.where(node_type==2)[0].tolist()

        region_nodes = x[region_list]
        patch_nodes = x[patch_list]
        thumbnail = x[tumbnail_list]
        patch_tree = tree[patch_list]

        n,c = patch_nodes.shape
        if n < self.last_pool_ratio*self.patch_ratio:
            patch_tree_values, patch_tree_counts = torch.unique(patch_tree, return_counts=True)
            value_add = []
            for i, value in enumerate(patch_tree_values):
                if patch_tree_counts[i]<4:
                    value_add += [int(value.item())]*int(4-patch_tree_counts[i].item())
            value_add = torch.tensor(value_add).to(patch_nodes.device)
            patch_tree = torch.cat((value_add, patch_tree)).long()
            e = torch.zeros((self.last_pool_ratio*self.patch_ratio-n,1024)).to(patch_nodes.device)
            patch_nodes = torch.cat((e,patch_nodes), dim=0)
        patch_nodes_ori = patch_nodes

        for mhit_ in self.mhit:
            region_nodes, patch_nodes = mhit_(
                region_nodes,
                patch_nodes,
                patch_tree.long()
            )

        # Fusion
        local_patch = self.ln(patch_nodes_ori+thumbnail)
        fusioned_patch_nodes = torch.mean(self.fusion(local_patch, patch_nodes), dim=0)

        # Classifier
        logits = self.classifier(fusioned_patch_nodes)
        prob = torch.sigmoid(logits)
 
        return prob.view(1,-1)


class Fusion_Block(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int
    ) -> None:
        super().__init__()
        self.conv_11 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 1 // 2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, local_patch: Tensor, global_patch: Tensor):
        patch_nodes = torch.cat((local_patch, global_patch), dim=0).unsqueeze(1).unsqueeze(0)
        patch_nodes = self.conv_11(patch_nodes).squeeze(0).squeeze(1)

        return patch_nodes


class H2GCN(nn.Module):
    def __init__(
            self,
            in_feats: int,
            n_hidden: int,
            out_feats: int,
            drop_out_ratio: float = 0.2,
            pool_ratio: list = [10],
    ):
        super(H2GCN, self).__init__()

        self.pool_ratio = pool_ratio

        convs_list = []
        pools_list = []
        for i, ratio in enumerate(pool_ratio):
            if i == 0:
                convs_list.append(RAConv(in_channels=in_feats, out_channels=n_hidden))
                pools_list.append(IHPool(in_channels=n_hidden, ratio=ratio, select="inter", dis="ou"))
            elif i == len(pool_ratio) - 1:
                convs_list.append(RAConv(in_channels=n_hidden, out_channels=out_feats))
                pools_list.append(IHPool(in_channels=out_feats, ratio=ratio, select="inter", dis="ou"))
            else:
                convs_list.append(RAConv(in_channels=n_hidden, out_channels=n_hidden))
                pools_list.append(IHPool(in_channels=n_hidden, ratio=ratio, select="inter", dis="ou"))

        self.convs = nn.Sequential(
            *convs_list
        )

        self.pools = nn.Sequential(
            *pools_list
        )

        self.norm = LayerNorm(in_feats)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out_ratio)

    def forward(self, data):
        x, edge_index, node_type, tree, x_y_index = data['x'], data["edge_index_tree_8nb"], data["node_type"], data[
            "node_tree"], data["x_y_index"]
        # x_y_index = x_y_index * 2 - 1
        x = self.norm(x)

        for i, _ in enumerate(self.pool_ratio):
            x = self.convs[i](x, edge_index, node_type)
            x = self.norm(x)
            x = self.dropout(x)
            x, edge_index, edge_weight, batch, cluster, node_type, tree, score, x_y_index = self.pools[i](x,
                                                                                                          edge_index,
                                                                                                          node_type=node_type,
                                                                                                          tree=tree,
                                                                                                          x_y_index=x_y_index)

            batch = edge_index.new_zeros(x.size(0))

        return x, edge_index, node_type, tree  # , x_y_index+1


class MobileHIT_Block(nn.Module):
    def __init__(
            self,
            channel: int = 1,
            re_patch_size: int = 1024,
            ori_patch_size: int = 1024,
            region_node_num: int = 5,
            patch_node_num: int = 20,
    ) -> None:
        super().__init__()

        self.re_patch_size = re_patch_size
        self.ori_patch_size = ori_patch_size
        self.region_node_num = region_node_num
        self.patch_node_num = patch_node_num

        # Patch Level:
        self.patch_channel = channel
        self.patch_block = MobileViTBlockV2(
            in_channels=self.patch_channel,
            re_patch_size=self.re_patch_size,
            ori_patch_size=self.ori_patch_size,
            node_num=patch_node_num,
        )

        # Region Level:
        self.region_channel = channel
        self.region_block = MobileViTBlockV2(
            in_channels=self.region_channel,
            re_patch_size=self.re_patch_size,
            ori_patch_size=self.ori_patch_size,
            node_num=region_node_num,
        )

        # SE
        self.se_region = SqueezeExcitation(input_c=region_node_num)
        self.se_patch = SqueezeExcitation(input_c=patch_node_num)

    def forward(
            self,
            region_nodes: Tensor,
            patch_nodes: Tensor,
            tree: Tensor):
        # Patch block:
        patch_nodes = patch_nodes.reshape(-1, self.re_patch_size).permute(1, 0).unsqueeze(0).unsqueeze(0)
        patch_nodes = self.patch_block(patch_nodes)
        patch_nodes = patch_nodes.squeeze(0).squeeze(0).permute(1, 0).reshape(-1, self.ori_patch_size)

        # Hierachical Interaction:
        region_patch_nodes = torch.cat([region_nodes[i - 1].unsqueeze(0) for i in tree], dim=0)
        patch_nodes = (patch_nodes + region_patch_nodes * self.se_patch(
            region_patch_nodes.unsqueeze(1).unsqueeze(0)).squeeze(0).squeeze(1)) / 2

        # Region block
        region_nodes = region_nodes.reshape(-1, self.re_patch_size).permute(1, 0).unsqueeze(0).unsqueeze(0)
        region_nodes = self.region_block(region_nodes)
        region_nodes = region_nodes.squeeze(0).squeeze(0).permute(1, 0).reshape(-1, self.ori_patch_size)

        return region_nodes, patch_nodes


import os
import torch
import time
import math, random
import numpy as np
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_scatter import scatter
from sklearn.cluster import KMeans
from torch_sparse import SparseTensor
from torch_geometric.nn import LEConv
from torch_geometric.utils import softmax
from scipy.spatial.distance import cdist
from typing import Union, Optional, Callable
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import remove_self_loops
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, sort_edge_index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


def euclidean_dist(x, y):
    # spatial distance
    x_xy = x[:, 0:2]
    y_xy = y[:, 0:2]

    m = x_xy.size(0)
    n = y_xy.size(0)
    e = x_xy.size(1)

    x1 = x_xy.unsqueeze(1).expand(m, n, e)
    y1 = y_xy.expand(m, n, e)
    dist_xy = (x1 - y1).pow(2).sum(2).float().sqrt()

    # fitness difference
    x_f = x[:, 2].unsqueeze(1)
    y_f = y[:, 2].unsqueeze(1)

    m = x_f.size(0)
    n = y_f.size(0)
    e = x_f.size(1)

    x2 = x_f.unsqueeze(1).expand(m, n, e)
    y2 = y_f.expand(m, n, e)
    dist_f = (x2 - y2).pow(2).sum(2).float().sqrt()

    return dist_xy + dist_f


class IHPool(torch.nn.Module):

    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.1,
                 GNN: Optional[Callable] = None, dropout: float = 0.0,
                 select='inter', dis='ou',
                 **kwargs):
        super(IHPool, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.dropout = dropout
        self.GNN = GNN
        self.select = select
        self.dis = dis

        self.lin = Linear(in_channels, in_channels)
        self.att = Linear(2 * in_channels, 1)
        self.gnn_score = LEConv(self.in_channels, 1)
        if self.GNN is not None:
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels, **kwargs)

        self.weight_1 = Parameter(torch.Tensor(1, in_channels))
        self.weight_2 = Parameter(torch.Tensor(1, in_channels))
        self.nonlinearity = torch.tanh

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.att.reset_parameters()
        self.gnn_score.reset_parameters()
        size = self.in_channels
        uniform(size, self.weight_1)
        uniform(size, self.weight_2)
        if self.GNN is not None:
            self.gnn_intra_cluster.reset_parameters()

    def forward(self, x, edge_index, node_type, tree, x_y_index, edge_weight=None, batch=None):
        r"""
        x : node feature;
        edge_index :  edge;
        node_type : the resolution-level of each node;
        tree : Correspondence between different level nodes;
        x_y_index : Space coordinates of each node;
        """

        N = x.size(0)
        N_1 = len(torch.where(node_type == 1)[0])
        N_2 = len(torch.where(node_type == 2)[0])

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=N)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        # fitness of second resolution-level node
        fitness_1 = (x[torch.where(node_type == 1)] * self.weight_1).sum(dim=-1)
        fitness_1 = self.nonlinearity(fitness_1 / self.weight_1.norm(p=2, dim=-1))

        # concat spatial information
        x_y_fitness_1 = torch.cat((x_y_index[torch.where(node_type == 1)], fitness_1.unsqueeze(1)), -1).to(device)
        sort_fitness_1, sort_fitness_1_index = torch.sort(fitness_1)

        # select nodes at intervals according to fitness
        if self.select == 'inter':
            if self.ratio < 1:
                index_of_threshold_fitness_1 = sort_fitness_1_index[
                    range(0, N_1, int(torch.ceil(torch.tensor(N_1 / (N_1 * self.ratio)))))].to(device)
            else:
                if N_1 < self.ratio:
                    index_of_threshold_fitness_1 = sort_fitness_1_index[
                        range(0, N_1, int(torch.ceil(torch.tensor(N_1 / N_1))))]
                else:
                    index_of_threshold_fitness_1 = sort_fitness_1_index[
                        range(0, N_1, int(torch.ceil(torch.tensor(N_1 / (self.ratio)))))]
        if len(index_of_threshold_fitness_1) < self.ratio:
            index_of_threshold_fitness_1 = torch.cat(
                (index_of_threshold_fitness_1, sort_fitness_1_index[-1].unsqueeze(0)))
        threshold_x_y_fitness_1 = x_y_fitness_1[index_of_threshold_fitness_1].to(device)

        # Clustering according to Euclidean distance
        if self.dis == 'ou':
            cosine_dis_1 = euclidean_dist(threshold_x_y_fitness_1, x_y_fitness_1).to(device)
            _, cosine_sort_index = torch.sort(cosine_dis_1, 0)
            cluster_1 = cosine_sort_index[0]

            # Calculate the coordinates of the nodes after clustering
        new_x_y_index_1 = scatter(x_y_index[torch.where(node_type == 1)], cluster_1, dim=0, reduce='mean').to(device)
        new_x_y_index = torch.cat((torch.tensor([[0, 0]]).to(device), new_x_y_index_1), 0).to(device)

        # fitness of third resolution-level node
        fitness_2 = (x[torch.where(node_type == 2)] * self.weight_2).sum(dim=-1).to(device)
        fitness_2 = self.nonlinearity(fitness_2 / self.weight_2.norm(p=2, dim=-1))

        # concat spatial information
        x_y_fitness_2 = torch.cat((x_y_index[torch.where(node_type == 2)], fitness_2.unsqueeze(1)), -1)
        x_y_index_2 = x_y_index[torch.where(node_type == 2)]

        # the nodes to be pooled are depend on the pooling results of corresponding low-resolution nodes
        cluster_2 = torch.tensor([0] * N_2, dtype=torch.long).to(device)
        cluster2_from_1 = cluster_1[tree[torch.where(node_type == 2)] - torch.min(tree[torch.where(node_type == 2)])]

        new_tree = torch.tensor([-1]).to(device)
        new_tree = torch.cat((new_tree, torch.tensor([0] * len(set(cluster_1.cpu().numpy()))).to(device)), 0).to(device)

        # Clustering of each substructure
        for k in range(len(set(cluster_1.cpu().numpy()))):
            # Get the index of each substructure
            index_of_after_cluster = torch.where(
                cluster2_from_1 == torch.tensor(sorted(list(set(cluster_1.cpu().numpy()))))[:, None][k].to(device))
            N_k = len(index_of_after_cluster[0])

            after_cluster_fitness_2 = fitness_2[index_of_after_cluster].to(device)
            after_cluster_x_y_fitness_2 = x_y_fitness_2[index_of_after_cluster].to(device)
            t_x_y_index_2 = x_y_index_2[index_of_after_cluster].to(device)

            sort_fitness_2, sort_fitness_2_index = torch.sort(after_cluster_fitness_2)

            # select nodes at intervals according to fitness
            if self.select == 'inter':
                if N_k < 4:
                    index_of_threshold_fitness_2 = sort_fitness_2_index[random.sample(range(N_k), N_k)].to(device)
                else:
                    index_of_threshold_fitness_2 = sort_fitness_2_index[random.sample(range(N_k), 4)].to(device)
            threshold_x_y_fitness_2 = after_cluster_x_y_fitness_2[index_of_threshold_fitness_2].to(device)

            # Clustering according to Euclidean distance
            if self.dis == 'ou':
                cosine_dis_2 = euclidean_dist(threshold_x_y_fitness_2, after_cluster_x_y_fitness_2).to(device)
                _, cosine_sort_index = torch.sort(cosine_dis_2, 0)
                t_cluster_2 = cosine_sort_index[0].to(device)

            new_x_y_index = torch.cat((new_x_y_index, scatter(t_x_y_index_2, t_cluster_2, dim=0, reduce='mean'))).to(
                device)
            t_cluster_2 += len(set(cluster_2.cpu().numpy())) * 2

            cluster_2[torch.where(cluster2_from_1 == k)] = t_cluster_2
            new_tree = torch.cat((new_tree, torch.tensor([k + 1] * len(set(t_cluster_2.cpu().numpy()))).to(device)))

        # Make the clustering results of different levels not repeated
        cluster = torch.tensor(range(N), dtype=torch.long).to(device)
        cluster[torch.where(node_type == 0)] = 0
        cluster[torch.where(node_type == 1)] = cluster_1 + 1
        cluster[torch.where(node_type == 2)] = cluster_2 + len(cluster_1) + 100

        # Remove invalid clusters
        cluster = torch.where(cluster[:, None] == torch.tensor(sorted(list(set(cluster.cpu().numpy())))).to(device))[
            -1].to(device)

        # new node's type
        node_type_0 = torch.tensor([0])
        node_type_1 = torch.tensor([1] * len(set(cluster_1.cpu().numpy())))
        node_type_2 = torch.tensor([2] * len(set(cluster_2.cpu().numpy())))
        node_type = torch.cat((node_type_0, node_type_1, node_type_2), 0).to(device)

        # X← S^T* X
        x = scatter(x, cluster, dim=0, reduce='mean')

        # A← S^T* A* S
        A = 0 * torch.ones((N, N)).to(device)
        A[edge_index[0], edge_index[1]] = 1
        A = scatter(A, cluster, dim=0, reduce='add')
        A = scatter(A, cluster, dim=1, reduce='add')
        row, col = torch.where(A != 0)
        edge_index = torch.stack([row, col], dim=0)

        batch = edge_index.new_zeros(x.size(0))
        fitness = torch.cat((torch.tensor([0]).to(device), fitness_1, fitness_2), 0)

        return x, edge_index, edge_weight, batch, cluster, node_type, new_tree, fitness, new_x_y_index

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__,
                                         self.in_channels, self.ratio)


import time
import math
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_scatter import scatter
from torch.nn import Parameter, Linear
from typing import Union, Tuple, Optional
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class RAConv(MessagePassing):
    _alpha: OptTensor
    t_alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = False, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(RAConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        if isinstance(in_channels, int):
            self.t_lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.t_lin_r = self.t_lin_l

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))
        self.t_att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.t_att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.t_alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.t_lin_l.weight)
        glorot(self.t_lin_r.weight)
        glorot(self.t_att_l)
        glorot(self.t_att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, node_type,
                size: Size = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        # create resolution-level graph
        start_node_type = node_type[edge_index[0]]
        start_x = x[edge_index[0]]

        new_index = start_node_type + edge_index[1] * 3

        t_x = scatter(start_x, new_index, dim=0, reduce='mean')
        t_x = torch.cat((x, t_x), 0)
        t_x_l = t_x_r = self.t_lin_l(t_x).view(-1, H, C)
        t_alpha_l = (t_x_l * self.t_att_l).sum(dim=-1)
        t_alpha_r = (t_x_r * self.t_att_r).sum(dim=-1)

        start = torch.tensor(sorted(list(set(new_index.cpu().numpy()))), dtype=torch.long) + len(node_type)
        end = torch.tensor(sorted(list(set(new_index.cpu().numpy()))), dtype=torch.long) // 3
        new_edge = torch.stack([start, end], dim=0).to(device)

        # resolution-level attention
        t_out = self.propagate(new_edge, x=(t_x_l, t_x_l), soft_index=None, type_edge=None, node_size=None,
                               alpha=(t_alpha_l, t_alpha_r), size=size)

        # node-level attention
        out = self.propagate(edge_index, x=(x_l, x_r), soft_index=new_index, type_edge=new_edge[0],
                             node_size=len(node_type),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None
        self.t_alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, soft_index: Tensor, type_edge, node_size,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:

        if self.t_alpha == None:
            alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, index, ptr, size_i)
            self.t_alpha = alpha
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        else:
            size_i = torch.max(soft_index) + 1

            alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, soft_index, ptr, size_i)
            self._alpha = alpha
            alpha = self.t_alpha[torch.where((type_edge - node_size) == soft_index[:, None])[-1]] + alpha
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


import numpy as np
from torch import nn, Tensor
from typing import Optional, Tuple, Union, Sequence
from torch.nn import functional as F


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        # print(scale.shape)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scores = F.hardsigmoid(scale, inplace=True)
        return scores


class MobileViTBlockV2(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ori_patch_size: int,
            re_patch_size: int,
            node_num: int,
            ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 1.0,
            n_attn_blocks: Optional[int] = 2,
            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[float] = 0.0,
            ffn_dropout: Optional[float] = 0.0,
            conv_ksize: Optional[int] = 3,
            attn_norm_layer: Optional[str] = "layer_norm_2d",
    ) -> None:
        super().__init__()

        self.transformer_in_dim = in_channels
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize

        self.global_rep = self._build_attn_layer(
            embed_channel=in_channels,
            ori_patch_size=ori_patch_size,
            re_patch_size=re_patch_size,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            node_num=node_num,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )
        self.ln = nn.LayerNorm(re_patch_size)

        self.layer_num = n_attn_blocks

    def _build_attn_layer(
            self,
            embed_channel: int,
            ffn_mult: Union[Sequence, int, float],
            n_layers: int,
            ori_patch_size: int,
            re_patch_size: int,
            node_num: int,
            attn_dropout: float,
            dropout: float,
            ffn_dropout: float,
            attn_norm_layer: str,
            *args,
            **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                    np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * embed_channel
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * embed_channel] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * embed_channel] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]
        '''
        Input: :math:`(B, C, P, N)` where :math:
        # `B` is batch size, 
        # `C` is input embedding dim,
        # `P` is number of pixels in a patch,
        # `N` is number of patches,
        '''
        global_rep = []

        for _ in range(n_layers):
            global_rep.append(
                LinearAttnFFN(
                    embed_channel=embed_channel,
                    ori_patch_size=ori_patch_size,
                    re_patch_size=re_patch_size,
                    node_num=node_num,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                )
            )

        return nn.Sequential(*global_rep)  # , d_model

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # learn global representations on all patches
        x = self.global_rep(x)
        return x


from typing import Optional, Union, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
    This layer can be used for self- as well as cross-attention.
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True
    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input
    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
            self,
            embed_channel: int,
            expand_channel: Optional[int] = 2,
            attn_dropout: Optional[float] = 0.0,
            bias: Optional[bool] = True,
            ori_patch_size: int = 1024,
            re_patch_size: int = 16,
            *args,
            **kwargs
    ) -> None:
        super().__init__()

        self.embed_channel = embed_channel
        self.ori_patch_size = ori_patch_size
        self.re_patch_size = re_patch_size

        if not expand_channel:
            self.expand_channel = self.embed_channel
        else:
            self.expand_channel = expand_channel

        self.qkv_proj = ConvLayer(
            in_channels=self.embed_channel,
            out_channels=1 + (2 * self.expand_channel),
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = ConvLayer(
            in_channels=self.expand_channel,
            out_channels=self.embed_channel,
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )

    def __repr__(self):
        return "{}(embed_channel={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_channel, self.attn_dropout.p
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.expand_channel, self.expand_channel], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        context_vector = context_vector.expand_as(value)

        out = F.relu(value) * context_vector
        out = self.out_proj(out)

        return out


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Optional[Union[int, Tuple[int, int]]] = 1,
            groups: Optional[int] = 1,
            bias: Optional[bool] = False,
            use_norm: Optional[bool] = True,
            use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2)
        )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class LinearAttnFFN(nn.Module):
    def __init__(
            self,
            # opts,
            embed_channel: int,
            ori_patch_size: int,
            re_patch_size: int,
            node_num: int,
            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[float] = 0.1,
            ffn_dropout: float = 0.1,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        self.re_patch_size = re_patch_size
        norm_dim = [ori_patch_size, node_num] if ori_patch_size == re_patch_size else [re_patch_size,
                                                                                       int(node_num * ori_patch_size / re_patch_size)]
        self.attn = nn.Sequential(
            nn.LayerNorm(norm_dim),
            LinearSelfAttention(
                embed_channel=embed_channel,
                ori_patch_size=ori_patch_size,
                attn_dropout=attn_dropout,
                re_patch_size=re_patch_size,
                bias=True,
            ),

        )
        self.attn_dropout = nn.Dropout(p=dropout)

        ln_dim = ori_patch_size

        ffn_latent_dim = embed_channel * 16

        self.ffn = nn.Sequential(
            ConvLayer(
                in_channels=embed_channel,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=True,
            ),
            nn.Dropout(p=ffn_dropout),
            ConvLayer(
                in_channels=ffn_latent_dim,
                out_channels=embed_channel,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=False,
            ),
            nn.Dropout(p=ffn_dropout),
        )

    def forward(
            self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        x_ = self.attn(x)
        x_ = self.attn_dropout(x_)
        x = x + x_
        x = x + self.ffn(x)

        return x