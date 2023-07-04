import torch
import torch.nn as nn
import torch.nn.functional as F

class MIL(nn.Module):
    def __init__(self, args):
        super(MIL, self).__init__()
        in_classes = args.in_classes
        out_classes = args.out_classes
        drop_out_ratio = args.drop_out_ratio
        self.lin = nn.Sequential(nn.Linear(in_classes, in_classes), nn.ReLU())
        self.q = nn.Sequential(nn.Linear(in_classes, in_classes), nn.Tanh())
        self.v = nn.Sequential(
            nn.Dropout(drop_out_ratio),
            nn.Linear(in_classes, in_classes)
        )

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(out_classes, out_classes, kernel_size=in_classes)
        self.fc = nn.Sequential(nn.Linear(in_classes, out_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c = self.fc(x)
        feats = x
        device = feats.device
        feats = self.lin(feats)
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)  # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        C = self.sigmoid(C)
        c = torch.max(c, 0, keepdim=True)[0]
        c = self.sigmoid(c)
        return c, C
