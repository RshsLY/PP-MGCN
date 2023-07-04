import torch.nn as nn
from torch_geometric.nn import LEConv
from typing import Union, Optional, Callable
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, sort_edge_index
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
from torch_geometric.nn import global_mean_pool,global_max_pool,GlobalAttention,dense_diff_pool,global_add_pool,TopKPooling,ASAPooling,SAGPooling
from torch_geometric.nn import GCNConv,ChebConv,SAGEConv,GraphConv,LEConv,LayerNorm,GATConv
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
        new_edge = torch.stack([start, end], dim=0).cuda()

        # resolution-level attention
        t_out = self.propagate(new_edge, x=(t_x_l, t_x_l), soft_index=None, type_edge=None, node_size=None,
                               alpha=(t_alpha_l, t_alpha_r), size=size)

        # node-level attention
        out = self.propagate(edge_index, x=(x_l, x_r), soft_index=new_index, type_edge=new_edge[0], node_size=len(node_type),
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
            alpha = self.t_alpha[torch.where((type_edge - node_size) == soft_index[:, None])[-1]] * alpha
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

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
        x_y_fitness_1 = torch.cat((x_y_index[torch.where(node_type == 1)], fitness_1.unsqueeze(1)), -1).cuda()
        sort_fitness_1, sort_fitness_1_index = torch.sort(fitness_1)

        # select nodes at intervals according to fitness
        if self.select == 'inter':
            if self.ratio < 1:
                index_of_threshold_fitness_1 = sort_fitness_1_index[range(0, N_1, int(torch.ceil(torch.tensor(N_1 / (N_1 * self.ratio)))))].cuda()
            else:
                if N_1 < self.ratio:
                    index_of_threshold_fitness_1 = sort_fitness_1_index[range(0, N_1, int(torch.ceil(torch.tensor(N_1 / N_1))))]
                else:
                    index_of_threshold_fitness_1 = sort_fitness_1_index[range(0, N_1, int(torch.ceil(torch.tensor(N_1 / (self.ratio)))))]

        threshold_x_y_fitness_1 = x_y_fitness_1[index_of_threshold_fitness_1].cuda()

        # Clustering according to Euclidean distance
        if self.dis == 'ou':
            cosine_dis_1 = euclidean_dist(threshold_x_y_fitness_1, x_y_fitness_1).cuda()
            _, cosine_sort_index = torch.sort(cosine_dis_1, 0)
            cluster_1 = cosine_sort_index[0]

            # Calculate the coordinates of the nodes after clustering
        new_x_y_index_1 = scatter(x_y_index[torch.where(node_type == 1)], cluster_1, dim=0, reduce='mean').cuda()
        new_x_y_index = torch.cat((torch.tensor([[0, 0]]).cuda(), new_x_y_index_1), 0).cuda()

        # fitness of third resolution-level node
        fitness_2 = (x[torch.where(node_type == 2)] * self.weight_2).sum(dim=-1).cuda()
        fitness_2 = self.nonlinearity(fitness_2 / self.weight_2.norm(p=2, dim=-1))
        # concat spatial information
        x_y_fitness_2 = torch.cat((x_y_index[torch.where(node_type == 2)], fitness_2.unsqueeze(1)), -1)
        x_y_index_2 = x_y_index[torch.where(node_type == 2)]

        # the nodes to be pooled are depend on the pooling results of corresponding low-resolution nodes
        cluster_2 = torch.tensor([0] * N_2, dtype=torch.long).cuda()
        cluster2_from_1 = cluster_1[tree[torch.where(node_type == 2)] - torch.min(tree[torch.where(node_type == 2)])]

        new_tree = torch.tensor([-1]).cuda()
        new_tree = torch.cat((new_tree, torch.tensor([0] * len(set(cluster_1.cpu().numpy()))).cuda()), 0).cuda()

        # Clustering of each substructure
        for k in range(len(set(cluster_1.cpu().numpy()))):
            # Get the index of each substructure
            index_of_after_cluster = torch.where(cluster2_from_1 == torch.tensor(sorted(list(set(cluster_1.cpu().numpy()))))[:, None][k].cuda())
            N_k = len(index_of_after_cluster[0])

            after_cluster_fitness_2 = fitness_2[index_of_after_cluster].cuda()
            after_cluster_x_y_fitness_2 = x_y_fitness_2[index_of_after_cluster].cuda()
            t_x_y_index_2 = x_y_index_2[index_of_after_cluster].cuda()

            sort_fitness_2, sort_fitness_2_index = torch.sort(after_cluster_fitness_2)

            # select nodes at intervals according to fitness
            if self.select == 'inter':
                if self.ratio < 1:
                    index_of_threshold_fitness_2 = sort_fitness_2_index[range(0, N_k, int(torch.ceil(torch.tensor(N_k / (N_k * self.ratio)))))].cuda()
                else:
                    if N_k == 1:
                        index_of_threshold_fitness_2 = sort_fitness_2_index[range(0, N_k, N_k)].cuda()
                    else:
                        index_of_threshold_fitness_2 = sort_fitness_2_index[range(0, N_k, N_k - 1)].cuda()
            threshold_x_y_fitness_2 = after_cluster_x_y_fitness_2[index_of_threshold_fitness_2].cuda()

            # Clustering according to Euclidean distance
            if self.dis == 'ou':
                cosine_dis_2 = euclidean_dist(threshold_x_y_fitness_2, after_cluster_x_y_fitness_2).cuda()
                _, cosine_sort_index = torch.sort(cosine_dis_2, 0)
                t_cluster_2 = cosine_sort_index[0].cuda()

            new_x_y_index = torch.cat((new_x_y_index, scatter(t_x_y_index_2, t_cluster_2, dim=0, reduce='mean'))).cuda()
            t_cluster_2 += len(set(cluster_2.cpu().numpy())) * 2

            cluster_2[torch.where(cluster2_from_1 == k)] = t_cluster_2
            new_tree = torch.cat((new_tree, torch.tensor([k + 1] * len(set(t_cluster_2.cpu().numpy()))).cuda()))

        # Make the clustering results of different levels not repeated
        cluster = torch.tensor(range(N), dtype=torch.long).cuda()
        cluster[torch.where(node_type == 0)] = 0
        cluster[torch.where(node_type == 1)] = cluster_1 + 1
        cluster[torch.where(node_type == 2)] = cluster_2 + len(cluster_1) + 100
        # Remove invalid clusters
        cluster = torch.where(cluster[:, None] == torch.tensor(sorted(list(set(cluster.cpu().numpy())))).cuda())[-1].cuda()

        # new node's type
        node_type_0 = torch.tensor([0])
        node_type_1 = torch.tensor([1] * len(set(cluster_1.cpu().numpy())))
        node_type_2 = torch.tensor([2] * len(set(cluster_2.cpu().numpy())))
        node_type = torch.cat((node_type_0, node_type_1, node_type_2), 0).cuda()

        # X← S^T* X
        x = scatter(x, cluster, dim=0, reduce='mean')

        # A← S^T* A* S
        A = 0 * torch.ones((N, N)).cuda()
        A[edge_index[0], edge_index[1]] = 1
        A = scatter(A, cluster, dim=0, reduce='add')
        A = scatter(A, cluster, dim=1, reduce='add')
        row, col = torch.where(A != 0)
        edge_index = torch.stack([row, col], dim=0)

        batch = edge_index.new_zeros(x.size(0))
        fitness = torch.cat((torch.tensor([0]).cuda(), fitness_1, fitness_2), 0)

        return x, edge_index, edge_weight, batch, cluster, node_type, new_tree, fitness, new_x_y_index

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__,
                                         self.in_channels, self.ratio)

class MIL(nn.Module):
    def __init__(self, args, drop_out_ratio=0.25, pool1_ratio=0.1, pool2_ratio=4, pool3_ratio=3, mpool_method="global_mean_pool"):
        in_feats=args.in_classes
        out_classes=512
        super(MIL, self).__init__()
        self.conv1 = RAConv(in_channels=in_feats, out_channels=out_classes)
        self.conv2 = RAConv(in_channels=out_classes, out_channels=out_classes)

        self.pool_1 = IHPool(in_channels=out_classes, ratio=pool1_ratio, select='inter', dis='ou')
        self.pool_2 = IHPool(in_channels=out_classes, ratio=pool2_ratio, select='inter', dis='ou')

        if mpool_method == "global_mean_pool":
            self.mpool = global_mean_pool
        elif mpool_method == "global_max_pool":
            self.mpool = global_max_pool
        elif mpool_method == "global_att_pool":
            att_net = nn.Sequential(nn.Linear(out_classes, out_classes // 2), nn.ReLU(), nn.Linear(out_classes // 2, 1))
            self.mpool = GlobalAttention(att_net)

        self.lin1 = torch.nn.Linear(out_classes, out_classes // 2)
        self.lin2 = torch.nn.Linear(out_classes // 2, args.out_classes)

        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=drop_out_ratio)
        self.softmax = nn.Softmax(dim=-1)
        self.norm0 = LayerNorm(in_feats)
        self.norm = LayerNorm(out_classes)
        self.norm1=LayerNorm(out_classes//2)

    def forward(self, data):
        x, batch = data["x"], data["batch"]
        edge_index, node_type, tree, x_y_index = data["edge_index_tree_8nb"], data["node_type"], data["node_tree"], data["x_y_index"]
        #x_y_index = x_y_index * 2 - 1

        x = self.norm0(x)

        x = self.conv1(x, edge_index, node_type)
        x = self.relu(x)
        x = self.norm(x)
        x = self.dropout(x)

        x, edge_index_1, edge_weight, batch, cluster_1, node_type_1, tree_1, score_1, x_y_index_1 = self.pool_1(x, edge_index, node_type=node_type, tree=tree, x_y_index=x_y_index)
        batch = edge_index_1.new_zeros(x.size(0))
        x1 = self.mpool(x, batch)

        x = self.conv2(x, edge_index_1, node_type_1)
        x = self.relu(x)
        x = self.norm(x)
        x = self.dropout(x)

        x, edge_index_2, edge_weight, batch, cluster_2, node_type_2, tree_2, score_2, x_y_index_2 = self.pool_2(x, edge_index_1, node_type=node_type_1, tree=tree_1, x_y_index=x_y_index_1)
        batch = edge_index_2.new_zeros(x.size(0))
        x2 = self.mpool(x, batch)

        x = x1 + x2

        x = self.lin1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)

        return x