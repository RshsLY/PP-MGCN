import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCNConv, GraphConv, GatedGraphConv, GATConv, SGConv, GINConv, GENConv, DeepGCNLayer


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class MIL(torch.nn.Module):
    def __init__(self, args,num_layers=4, edge_agg='spatial', multires=False, resample=0,
                 fusion=None, num_features=1024, hidden_dim=512, linear_dim=64, use_edges=False, pool=False, dropout=0.25):
        super(MIL, self).__init__()
        in_classes = args.in_classes
        out_classes = args.out_classes
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers-1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(in_classes, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(in_classes, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim * (self.num_layers + 1), hidden_dim * (self.num_layers + 1),), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim * (self.num_layers + 1), D=hidden_dim *(self.num_layers + 1), dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim * (self.num_layers + 1), hidden_dim * (self.num_layers + 1)), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = torch.nn.Linear(hidden_dim * (self.num_layers + 1), out_classes)

    def forward(self, x,edge_index):

        edge_attr = None
        x = self.fc(x)
        x_ = x

        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)

        h_path = x_
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path).squeeze()
        logits = self.classifier(h).unsqueeze(0)  # logits needs to be a [1 x 4] vector
        hazards = torch.sigmoid(logits)

        return hazards