import random

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class AddGraphConv(MessagePassing):
    def __init__(self, in_class, out_class):
        super().__init__(aggr="mean")
        self.lin = nn.Linear(in_class, out_class)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return self.lin(aggr_out)


class MaskAddGraphConv(MessagePassing):
    def __init__(self, in_class, out_class):
        super().__init__(aggr="mean")
        self.lin = nn.Linear(in_class, out_class)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        if random.random() > 0.6:
            return x_j
        return torch.zeros_like(x_j).cuda()

    def update(self, aggr_out):
        return self.lin(aggr_out)
