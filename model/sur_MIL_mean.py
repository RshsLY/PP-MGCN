import torch.nn as nn
import torch
class MIL(nn.Module):
    def __init__(self, args):
        super(MIL, self).__init__()
        in_classes=args.in_classes
        out_classes = args.out_classes
        drop_out_ratio=args.drop_out_ratio
        self.l0 = torch.nn.Linear(in_classes, in_classes//2)
        in_classes = in_classes // 2
        self.l1 = torch.nn.Linear(in_classes, out_classes)
        self.relu = torch.nn.ReLU(True)
        self.drop = nn.Dropout(drop_out_ratio)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l0(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.l1(x)
        x = torch.mean(x, 0, keepdim=True)
        x = self.sigmoid(x)
        return x