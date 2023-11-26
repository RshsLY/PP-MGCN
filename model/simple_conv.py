import random

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MaskAddGraphConv(MessagePassing):
    def __init__(self, in_class, out_class, ):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_class, out_class)
        self.mask_prob = 0

    def forward(self, x, edge_index, mask_prob):
        self.mask_prob=mask_prob
        return self.propagate(edge_index, x=self.lin(x))

    def message(self, x_j):
        if random.random() >= self.mask_prob:
            return x_j
        return torch.zeros_like(x_j).cuda()

    def update(self, aggr_out):
        return aggr_out
    # return self.lin(aggr_out)
