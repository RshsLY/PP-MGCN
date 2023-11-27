import copy
import time

import numpy as np
import torch.nn as nn
import  torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GATv2Conv,SAGPooling,global_mean_pool,ASAPooling,global_max_pool,GCNConv,InstanceNorm,GINConv,GENConv,DeepGCNLayer
from torch.nn import   LeakyReLU,LayerNorm

from model.simple_conv import MaskAddGraphConv


class MIL(nn.Module):
    def __init__(self, args):
        super(MIL, self).__init__()
        in_classes=args.in_classes
        out_classes = args.out_classes
        drop_out_ratio=args.drop_out_ratio
        self.number_scale = args.number_scale
        self.using_Swin=args.using_Swin
        self.gcn_layer = args.gcn_layer
        self.magnification_scale=args.magnification_scale
        self.l0 = nn.Sequential(
            nn.Linear(in_classes, in_classes//2),
            nn.LeakyReLU(),
            nn.Dropout(drop_out_ratio)
        )
        in_classes=in_classes//2

        self.g0= torch.nn.ModuleList()
        self.g1 = torch.nn.ModuleList()
        self.g2 = torch.nn.ModuleList()
        self.g3 = torch.nn.ModuleList()
        self.g4 = torch.nn.ModuleList()
        self.gnn_convs= [self.g0, self.g1,self.g2,self.g3,self.g4]
        self.gnn_convs_diff= torch.nn.ModuleList()

        self.att1 = torch.nn.ModuleList()
        self.att2=torch.nn.ModuleList()
        self.att3=torch.nn.ModuleList()
        self.att_softmax=torch.nn.ModuleList()
        self.att_l1=torch.nn.ModuleList()
        #self.trans = torch.nn.ModuleList()
        for i in range (self.number_scale):
            for j in range(self.gcn_layer):
                self.gnn_convs[i].append(DeepGCNLayer(MaskAddGraphConv(in_classes, in_classes),
                                         LayerNorm(in_classes),
                                         LeakyReLU(), block='plain', dropout=0.1,ckpt_grad=0))

            self.gnn_convs_diff.append(DeepGCNLayer(MaskAddGraphConv(in_classes, in_classes),
                                         LayerNorm(in_classes),
                                         LeakyReLU(), block='plain', dropout=0.1,ckpt_grad=0))
            self.att1.append(nn.Sequential(nn.Linear(in_classes*(self.gcn_layer+1), in_classes*(self.gcn_layer+1)), nn.Tanh(), nn.Dropout(drop_out_ratio),))
            self.att2.append(nn.Sequential( nn.Linear(in_classes*(self.gcn_layer+1), in_classes*(self.gcn_layer+1)),nn.Sigmoid(),nn.Dropout(drop_out_ratio),))
            self.att3.append(nn.Linear(in_classes*(self.gcn_layer+1) , 1))
            self.att_softmax.append(nn.Softmax(dim=-1))
            self.att_l1.append(nn.Sequential(
                nn.Linear(in_classes*(self.gcn_layer+1), in_classes*(self.gcn_layer+1)),
                nn.LeakyReLU(),
                nn.Dropout(drop_out_ratio),
            ))

            #self.trans.append(nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=in_classes, nhead=8), num_layers=2))


        self.l_last=nn.Sequential(
                nn.Linear(in_classes*(self.number_scale)*(self.gcn_layer+1), in_classes*(self.number_scale)*(self.gcn_layer+1)),
                nn.LeakyReLU(),
                nn.Dropout(drop_out_ratio),
            )
        self.l_cla =  nn.Sequential(torch.nn.Linear(in_classes*(self.number_scale)*(self.gcn_layer+1), out_classes),nn.Sigmoid() )




    def forward(self, x,edge_index,edge_index_diff,feats_size_list,mask_prob):
        # tim=time.time()
        x = self.l0(x)
        pssz = [0, 0, 0, 0,0]
        if self.using_Swin ==1:
            bag_count_sigle_layer=1
            bag_count_idx=0
            for i in range(self.number_scale):
                for j in range(bag_count_sigle_layer):
                    pssz[i]=pssz[i]+feats_size_list[bag_count_idx]
                    bag_count_idx+=1
                if i<=0:
                    bag_count_sigle_layer=bag_count_sigle_layer*self.magnification_scale*self.magnification_scale
        else:
            for i in range(len(feats_size_list)):
                pssz[i]=feats_size_list[i]
        rm_x_count=0
        all_x_count=x.shape[0]
        x_=[]
        for i in range(self.number_scale):
            x = torch.split(x, [rm_x_count,pssz[i],all_x_count-rm_x_count-pssz[i]], 0)
            xx = x[1]
            edge_index[i] = edge_index[i] - rm_x_count

            x_.append(xx)
            for conv in self.gnn_convs[i]:
                xx = conv(xx, edge_index[i],0)
                x_[-1] = torch.cat((x_[-1], xx), dim=-1)
            # xx = torch.unsqueeze(xx, 0)
            # xx = self.trans[i](xx)
            # xx = torch.squeeze(xx, 0)
            x = torch.cat((x[0], xx, x[2]))
            edge_index[i] = edge_index[i] + rm_x_count

            if i != (self.number_scale - 1):
                x = torch.split(x,[rm_x_count, pssz[i] + pssz[i + 1], all_x_count - rm_x_count - pssz[i] - pssz[i + 1]],0)
                xx = x[1]
                edge_index_diff[i] = edge_index_diff[i] - rm_x_count
                xx = self.gnn_convs_diff[i](xx, edge_index_diff[i], mask_prob)
                x = torch.cat((x[0], xx, x[2]))
                edge_index_diff[i] = edge_index_diff[i] + rm_x_count
                rm_x_count=rm_x_count+pssz[i]


        predict_list=[]
        x_v_list=[]
        at_=[]


        for i in range (self.number_scale):
            x_sub=x_[i]
            x_sub = self.att_l1[i](x_sub)
            at1 = self.att1[i](x_sub)
            at2 = self.att2[i](x_sub)
            a = at1.mul(at2)
            a = self.att3[i](a)
            a = torch.transpose(a, 0, 1)  # 1*N
            a = self.att_softmax[i](a)
            at_.append(a)
            x_sub = torch.mm(a, x_sub)
            x_v_list.append(x_sub)

        x_v=x_v_list[0]
        for i in range(len(x_v_list)):
            if i==0:continue
            x_v=torch.cat((x_v,x_v_list[i]),dim=1)

        x_v=self.l_last(x_v)
        x_v=self.l_cla(x_v)
        predict_list.append(x_v)
        # print("INnet:",time.time()-tim)
        return predict_list,at_