import torch.nn as nn
import  torch
class MIL(nn.Module):
    def __init__(self, args):
        super(MIL, self).__init__()
        in_classes=args.in_classes
        out_classes = args.out_classes
        drop_out_ratio=args.drop_out_ratio
        self.l0 = nn.Sequential(
            nn.Linear(in_classes, in_classes // 2),
            nn.ReLU(),
            nn.Dropout(drop_out_ratio)
        )
        in_classes=in_classes//2
        self.att1=nn.Sequential(
            nn.Linear(in_classes, in_classes),
            nn.Tanh(),
        )
        self.att2 = nn.Sequential(
            nn.Linear(in_classes, in_classes ),
            nn.Sigmoid(),
        )
        self.att3=nn.Linear(in_classes, 1)

        self.l1 = torch.nn.Linear(in_classes, out_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()


    def forward(self, x):
        x = self.l0(x)

        at1 = self.att1(x)
        at2 = self.att2(x)
        a=at1.mul(at2)
        a=self.att3(a)
        a= torch.transpose(a,0,1) #1*N
        a= self.softmax(a)
        x=torch.mm(a,x)
        x = self.l1(x)
        x = self.sigmoid(x)
        return x
