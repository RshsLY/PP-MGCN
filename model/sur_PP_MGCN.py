
import torch.nn as nn

import model.sur_SWAP_GCN


class MIL(nn.Module):
    def __init__(self, args,args_cmp):
        super(MIL, self).__init__()
        self.basic_flow=model.sur_SWAP_GCN.MIL(args)
        self.contrastive_flow = model.sur_SWAP_GCN.MIL(args_cmp)


    def forward(self, x,edge_index,edge_index_diff,feats_size_list,x_cmp,edge_index_cmp,edge_index_diff_cmp,feats_size_list_cmp,mask_prob):
        predict_list, at_=self.basic_flow(x,edge_index,edge_index_diff,feats_size_list,0)
        predict_list_cmp, at_cmp = self.contrastive_flow(x_cmp, edge_index_cmp, edge_index_diff_cmp, feats_size_list_cmp, 0)
        return predict_list,at_,predict_list_cmp,at_cmp