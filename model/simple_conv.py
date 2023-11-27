import random
import time

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
        if self.mask_prob==0:
            return x_j
        mask_=torch.rand(x_j.shape[0]).cuda()
        mask_=torch.where(mask_>self.mask_prob,torch.ones(x_j.shape[0]).cuda(),torch.zeros(x_j.shape[0]).cuda())
        mask_=torch.unsqueeze(mask_,-1)
        mask_=torch.repeat_interleave(mask_,x_j.shape[1],-1)
        x_j=mask_.cuda()*x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out
    # return self.lin(aggr_out)
