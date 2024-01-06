import copy
import math
import os
import time

import pandas
import numpy as np
import torch
def get_bag(args,WSI_name, sur_time,censor,data_map):

    csv_path = ""
    if args.dataset == 'TCGA_LUAD':
        csv_path = os.path.join('TCGA_LUAD_Feats', WSI_name + '_'+str(args.patch_size)+'_0_0' + '.csv')
    if args.dataset == 'TCGA_LUSC':
        csv_path = os.path.join('TCGA_LUSC_Feats', WSI_name + '_'+str(args.patch_size)+'_0_0' + '.csv')
    if args.dataset == 'TCGA_UCEC':
        csv_path = os.path.join('TCGA_UCEC_Feats', WSI_name + '_'+str(args.patch_size)+'_0_0' + '.csv')
    if args.dataset == 'TCGA_BRCA':
        csv_path = os.path.join('TCGA_BRCA_Feats', WSI_name + '_' + str(args.patch_size) + '_0_0' + '.csv')
    if args.dataset == 'TCGA_BLCA':
        csv_path = os.path.join('TCGA_BLCA_Feats', WSI_name + '_' + str(args.patch_size) + '_0_0' + '.csv')
    if args.dataset == 'TCGA_GBMLGG':
        csv_path = os.path.join('TCGA_GBMLGG_Feats', WSI_name + '_' + str(args.patch_size) + '_0_0' + '.csv')
    if args.model=='sur_SWAP_GCN' or args.model=='sur_PP_MGCN':
        #for cmp model
        WSI_name=WSI_name+"_scale"+str(args.number_scale)

        if data_map.__contains__(WSI_name)==False:
            #tim=time.time()
            feats_mp = {}
            feats_list = []
            feats_info=[]
            patch_size_list=[args.patch_size]
            for i in range(args.number_scale-1):
                patch_size_list.append(patch_size_list[patch_size_list.__len__()-1]*args.magnification_scale)
            feats_size_list=[]
            feats_count=0
            min_ps=patch_size_list[0]
            max_row=0
            max_col=0
            df = pandas.read_csv(csv_path)
            for i in range(0, df.shape[1]):
                pos_str = df.columns[i]
                top_row = int(pos_str.split(',')[0])
                top_col = int(pos_str.split(',')[1])
                max_row=max(max_row,top_row+min_ps+1)
                max_col=max(max_col,top_col+min_ps+1)
                feats_list.append(df[df.columns[i]].values)
                feats_info.append((min_ps,top_row,top_col))
                feats_mp[(min_ps, 0, 0, top_row, top_col)] = feats_count
                feats_count = feats_count + 1
            feats_size_list.append(feats_count)

            for now_ps in patch_size_list:
                if now_ps==min_ps:
                    continue
                # for start_row in range(-now_ps + now_ps//args.magnification_scale, 1, now_ps//args.magnification_scale):
                #     for start_col in range(-now_ps + now_ps//args.magnification_scale, 1, now_ps//args.magnification_scale):
                for start_row in range(-now_ps + min_ps, 1, min_ps):
                    for start_col in range(-now_ps + min_ps, 1, min_ps):
                        if args.using_Swin == 0  and (start_col!=0 or start_row!=0):
                            continue
                        feats_count_now=0
                        for top_row in range(start_row, max_row, now_ps):
                            for top_col in range(start_col, max_col, now_ps):
                                flag_sub=0
                                for SR in range(-now_ps // args.magnification_scale + min_ps, 1, min_ps):
                                    for SC in range(-now_ps // args.magnification_scale + min_ps, 1, min_ps):
                                        for TR in range(top_row, top_row + now_ps, now_ps//args.magnification_scale):
                                            for TC in range(top_col, top_col + now_ps, now_ps//args.magnification_scale):
                                                idx = (now_ps // args.magnification_scale, SR, SC, TR, TC)
                                                if  flag_sub==0 and feats_mp.__contains__(idx)  \
                                                        and TR+now_ps//args.magnification_scale <= top_row + now_ps and TC+now_ps//args.magnification_scale <= top_col +now_ps:
                                                    flag_sub+=1
                                #print(now_ps,'-',flag_sub)
                                if flag_sub>=1:
                                    feats_list.append([0]*1024)
                                    feats_info.append((now_ps,top_row,top_col))
                                    feats_mp[(now_ps, start_row, start_col, top_row, top_col)] = feats_count
                                    feats_count=feats_count+1
                                    feats_count_now = feats_count_now + 1
                        feats_size_list.append(feats_count_now)

            edge_index=[[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
            edge_index_diff=[[[],[]],[[],[]],[[],[]],[[],[]]]
            for i in feats_mp:
                now_ps = i[0]
                start_row = i[1]
                start_col = i[2]
                top_row = i[3]
                top_col = i[4]

                #self loop
                for sz_idx in range(patch_size_list.__len__()):
                    if patch_size_list[sz_idx] == now_ps:
                        edge_index[sz_idx][0].append(feats_mp[i])
                        edge_index[sz_idx][1].append(feats_mp[i])

                poe_ne_list = [(top_row + now_ps, top_col), (top_row - now_ps, top_col),
                               (top_row, top_col + now_ps), (top_row, top_col - now_ps),
                               (top_row + now_ps, top_col + now_ps),
                               (top_row - now_ps, top_col + now_ps),
                               (top_row + now_ps, top_col - now_ps),
                               (top_row - now_ps, top_col - now_ps)]
                for j in poe_ne_list:
                    idx = (now_ps, start_row, start_col, j[0], j[1])
                    if feats_mp.__contains__(idx):
                        for sz_idx in range(patch_size_list.__len__()):
                            if patch_size_list[sz_idx]== now_ps:
                                edge_index[sz_idx][0].append(feats_mp[idx])
                                edge_index[sz_idx][1].append(feats_mp[i])
                for SR in range(-now_ps // args.magnification_scale + min_ps, 1, min_ps):
                    for SC in range(-now_ps // args.magnification_scale + min_ps, 1, min_ps):
                        for TR in range(top_row, top_row + now_ps, now_ps//args.magnification_scale):
                            for TC in range(top_col, top_col + now_ps, now_ps//args.magnification_scale):
                                idx = (now_ps // args.magnification_scale, SR, SC, TR, TC)
                                if  feats_mp.__contains__(idx) and TR+now_ps//args.magnification_scale <= top_row + now_ps and TC+now_ps//args.magnification_scale <= top_col + now_ps:
                                    for sz_idx in range(patch_size_list.__len__()):
                                        if patch_size_list[sz_idx] == now_ps:
                                            edge_index_diff[sz_idx-1][1].append(feats_mp[idx])
                                            edge_index_diff[sz_idx-1][0].append(feats_mp[i])
                                            edge_index_diff[sz_idx-1][0].append(feats_mp[idx])
                                            edge_index_diff[sz_idx-1][1].append(feats_mp[i])
            feats_list = np.array(feats_list)
            feats_list = torch.from_numpy(feats_list)
            feats_list = feats_list.to(torch.float32)
            for idx in range(edge_index.__len__()):
                edge_index[idx] = np.array(edge_index[idx])
                edge_index[idx]= torch.from_numpy(edge_index[idx])
                edge_index[idx]= edge_index[idx].to(torch.long)
            for idx in range(edge_index_diff.__len__()):
                edge_index_diff[idx] = np.array(edge_index_diff[idx])
                edge_index_diff[idx] = torch.from_numpy(edge_index_diff[idx])
                edge_index_diff[idx] = edge_index_diff[idx].to(torch.long)
            sur_time = np.array(sur_time)
            sur_time = torch.from_numpy(sur_time)
            sur_time = sur_time.to(torch.long)
            censor = np.array(censor)
            censor = torch.from_numpy(censor)
            censor = censor.to(torch.long)
            data_map[WSI_name]={"feats_list":feats_list,"sur_time":sur_time,"censor":censor,"edge_index":edge_index,"edge_index_diff":edge_index_diff,
                                "feats_size_list":feats_size_list,"feats_info":feats_info}
            #print("building:", time.time() - tim)
        #tim=time.time()
        feats_list=copy.deepcopy(data_map[WSI_name]["feats_list"]).cuda()
        sur_time=copy.deepcopy(data_map[WSI_name]["sur_time"]).cuda()
        censor=copy.deepcopy(data_map[WSI_name]["censor"]).cuda()
        edge_index=copy.deepcopy(data_map[WSI_name]["edge_index"])
        edge_index_diff = copy.deepcopy(data_map[WSI_name]["edge_index_diff"])
        for i in range(len(edge_index)):
            edge_index[i]=edge_index[i].cuda()
        for i in range(len(edge_index_diff)):
            edge_index_diff[i] = edge_index_diff[i].cuda()
        feats_size_list=copy.deepcopy(data_map[WSI_name]["feats_size_list"])
        feats_info=copy.deepcopy(data_map[WSI_name]["feats_info"])
        #print("mapping:",time.time()-tim)
        return feats_list,sur_time,censor,edge_index,edge_index_diff,feats_size_list,feats_info

    elif args.model == 'sur_Patch_GCN':
        if data_map.__contains__(WSI_name) == False:
            df = pandas.read_csv(csv_path)
            feats_mp = {}
            feats_list = []
            for i in range(0, df.shape[1]):
                feats_mp[df.columns[i]] = i
                feats_list.append(df[df.columns[i]].values)
            chu = []
            ru = []
            for i in range(0, df.shape[1]):
                pos_str = df.columns[i]
                row = int(pos_str.split(',')[0])
                col = int(pos_str.split(',')[1])
                poe_ne_str_list = [str(row + args.patch_size) + ',' + str(col), str(row - args.patch_size) + ',' + str(col),
                                   str(row) + ',' + str(col + args.patch_size), str(row) + ',' + str(col - args.patch_size),
                                   str(row + args.patch_size) + ',' + str(col + args.patch_size), str(row + args.patch_size) + ',' + str(col - args.patch_size),
                                   str(row - args.patch_size) + ',' + str(col + args.patch_size), str(row - args.patch_size) + ',' + str(col - args.patch_size), ]
                for pos_ne_str in (poe_ne_str_list):
                    if feats_mp.__contains__(pos_ne_str):
                        chu.append(i)
                        ru.append(feats_mp[pos_ne_str])
            feats_list = np.array(feats_list)
            feats_list = torch.from_numpy(feats_list)
            feats_list = feats_list.to(torch.float32)
            edge_index = np.array([chu, ru])
            edge_index = torch.from_numpy(edge_index)
            edge_index = edge_index.to(torch.long)
            sur_time = np.array(sur_time)
            sur_time = torch.from_numpy(sur_time)
            sur_time = sur_time.to(torch.long)
            censor = np.array(censor)
            censor = torch.from_numpy(censor)
            censor = censor.to(torch.long)
            data_map[WSI_name] = {"feats_list": feats_list, "sur_time": sur_time, "censor": censor, "edge_index": edge_index}
        #tim
        feats_list = copy.deepcopy(data_map[WSI_name]["feats_list"]).cuda()
        sur_time = copy.deepcopy(data_map[WSI_name]["sur_time"]).cuda()
        censor = copy.deepcopy(data_map[WSI_name]["censor"]).cuda()
        edge_index = copy.deepcopy(data_map[WSI_name]["edge_index"]).cuda()
        return feats_list, sur_time,censor,edge_index

    elif args.model=='sur_H2_MIL' or args.model == 'sur_HIGT':
        if data_map.__contains__(WSI_name) == False:
            # tim=time.time()
            feats_mp = {}
            feats_mp_idx={}
            feats_list = []
            patch_size_list = [args.patch_size,args.patch_size*2,1000000]
            feats_size_list = []
            feats_count = 0
            min_ps = patch_size_list[0]
            max_row = 0
            max_col = 0
            df = pandas.read_csv(csv_path)
            for i in range(0, df.shape[1]):
                pos_str = df.columns[i]
                top_row = int(pos_str.split(',')[0])
                top_col = int(pos_str.split(',')[1])
                max_row = max(max_row, top_row + min_ps + 1)
                max_col = max(max_col, top_col + min_ps + 1)
                feats_list.append(df[df.columns[i]].values)
                feats_mp[(min_ps, 0, 0, top_row, top_col)] = feats_count
                feats_mp_idx[feats_count]=(min_ps, 0, 0, top_row, top_col)
                feats_count = feats_count + 1
            feats_size_list.append(feats_count)

            for now_ps in patch_size_list:
                if now_ps != patch_size_list[1]:
                    continue
                feats_count_now = 0
                for top_row in range(0, max_row, now_ps):
                    for top_col in range(0, max_col, now_ps):
                        flag_sub = 0
                        for SR in range(-now_ps // 2 + min_ps, 1, min_ps):
                            for SC in range(-now_ps // 2 + min_ps, 1, min_ps):
                                for TR in range(top_row, top_row + now_ps, min_ps):
                                    for TC in range(top_col, top_col + now_ps, min_ps):
                                        idx = (now_ps // 2, SR, SC, TR, TC)
                                        if flag_sub == 0 and feats_mp.__contains__(idx) and TR + now_ps // 2 <= top_row + now_ps and TC + now_ps // 2 <= top_col + now_ps:
                                            flag_sub += 1
                        # print(now_ps,'-',flag_sub)
                        if flag_sub >= 1:
                            feats_list.append([0] * 1024)
                            feats_mp[(now_ps, 0, 0, top_row, top_col)] = feats_count
                            feats_mp_idx[feats_count]=(now_ps, 0, 0, top_row, top_col)
                            feats_count = feats_count + 1
                            feats_count_now = feats_count_now + 1
                feats_size_list.append(feats_count_now)

            feats_list_new=[]
            feats_mp_new={}
            batch = [0] * (feats_count+1)
            edge_index = [[], []]
            node_tree = []
            node_type = []
            dd=[]
            feats_list_new.append([0] * 1024)
            feats_mp_new[(1000000,0,0,0,0)]=0
            node_tree.append(-1)
            node_type.append(0)
            dd.append([0,0])
            feats_size_list.append(1)
            for i in range(len(feats_list)):
                idd=len(feats_list)-1-i
                feats_list_new.append(feats_list[idd])
                feats_mp_new[feats_mp_idx[idd]]=len(feats_mp_new)
                node_tree.append(0)
                node_type.append(1)
                dd.append([feats_mp_idx[idd][3]/(max_row-1),feats_mp_idx[idd][4]/(max_col-1)])
                #print([feats_mp_idx[idd][3]/max_row,feats_mp_idx[idd][4]/max_col],end=" ")
            patch_size_list.reverse()


            for i in feats_mp_new:
                now_ps = i[0]
                start_row = i[1]
                start_col = i[2]
                top_row = i[3]
                top_col = i[4]
                poe_ne_list = [(top_row + now_ps, top_col), (top_row - now_ps, top_col),
                               (top_row, top_col + now_ps), (top_row, top_col - now_ps),
                               (top_row + now_ps, top_col + now_ps),
                               (top_row - now_ps, top_col + now_ps),
                               (top_row + now_ps, top_col - now_ps),
                               (top_row - now_ps, top_col - now_ps)]
                for j in poe_ne_list:
                    idx = (now_ps, start_row, start_col, j[0], j[1])
                    if feats_mp_new.__contains__(idx):
                            edge_index[0].append(feats_mp_new[idx])
                            edge_index[1].append(feats_mp_new[i])
                if now_ps==patch_size_list[1]:
                    edge_index[0].append(0)
                    edge_index[1].append(feats_mp_new[i])
                    edge_index[1].append(0)
                    edge_index[0].append(feats_mp_new[i])
                    for SR in range(-now_ps // 2 + min_ps, 1, min_ps):
                        for SC in range(-now_ps // 2 + min_ps, 1, min_ps):
                            for TR in range(top_row, top_row + now_ps, min_ps):
                                for TC in range(top_col, top_col + now_ps, min_ps):
                                    idx = (now_ps // 2, SR, SC, TR, TC)
                                    if feats_mp_new.__contains__(idx) and TR + now_ps // 2 <= top_row + now_ps and TC + now_ps // 2 <= top_col + now_ps:
                                        edge_index[0].append(feats_mp_new[idx])
                                        edge_index[1].append(feats_mp_new[i])
                                        edge_index[1].append(feats_mp_new[idx])
                                        edge_index[0].append(feats_mp_new[i])
                                        node_tree[feats_mp_new[idx]]=feats_mp_new[i]
                                        node_type[feats_mp_new[idx]]=2
            batch = np.array(batch)
            batch = torch.from_numpy(batch)
            batch = batch.to(torch.long)
            dd = np.array(dd)
            dd = torch.from_numpy(dd)
            dd = dd.to(torch.float32)
            node_tree = np.array(node_tree)
            node_tree = torch.from_numpy(node_tree)
            node_tree = node_tree.to(torch.long)
            node_type = np.array(node_type)
            node_type = torch.from_numpy(node_type)
            node_type = node_type.to(torch.long)
            feats_list_new = np.array(feats_list_new)
            feats_list_new = torch.from_numpy(feats_list_new)
            feats_list_new = feats_list_new.to(torch.float32)
            edge_index = np.array(edge_index)
            edge_index= torch.from_numpy(edge_index)
            edge_index = edge_index.to(torch.long)
            sur_time = np.array(sur_time)
            sur_time = torch.from_numpy(sur_time)
            sur_time = sur_time.to(torch.long)
            censor = np.array(censor)
            censor = torch.from_numpy(censor)
            censor = censor.to(torch.long)

            data_map[WSI_name] = {"batch": batch, "sur_time": sur_time, "censor": censor, "edge_index_tree_8nb": edge_index,
                                  "node_tree": node_tree,"node_type":node_type,"x":feats_list_new,"x_y_index":dd}
            # print("building:", time.time() - tim)
            # tim=time.time()
        feats_list_new = copy.deepcopy(data_map[WSI_name]["x"]).cuda()
        sur_time = copy.deepcopy(data_map[WSI_name]["sur_time"]).cuda()
        censor = copy.deepcopy(data_map[WSI_name]["censor"]).cuda()
        edge_index = copy.deepcopy(data_map[WSI_name]["edge_index_tree_8nb"]).cuda()
        batch = copy.deepcopy(data_map[WSI_name]["batch"]).cuda()
        node_tree = copy.deepcopy(data_map[WSI_name]["node_tree"]).cuda()
        node_type = copy.deepcopy(data_map[WSI_name]["node_type"]).cuda()
        dd = copy.deepcopy(data_map[WSI_name]["x_y_index"]).cuda()
        # print("mapping:",time.time()-tim)
        return  {"batch": batch, "sur_time": sur_time, "censor": censor, "edge_index_tree_8nb": edge_index,
                                  "node_tree": node_tree,"node_type":node_type,"x":feats_list_new,"x_y_index":dd}

    elif args.model=='sur_MIL_Trans':
        if data_map.__contains__(WSI_name) == False:
            # tim=time.time()

            feats_mp = {}
            feats_list = []
            feats_info = []
            patch_size_list = [args.patch_size]
            for i in range(args.number_scale - 1):
                patch_size_list.append(patch_size_list[patch_size_list.__len__() - 1] * args.magnification_scale)
            feats_size_list = []
            feats_count = 0
            min_ps = patch_size_list[0]
            max_row = 0
            max_col = 0
            df = pandas.read_csv(csv_path)
            for i in range(0, df.shape[1]):
                pos_str = df.columns[i]
                top_row = int(pos_str.split(',')[0])
                top_col = int(pos_str.split(',')[1])
                max_row = max(max_row, top_row + min_ps + 1)
                max_col = max(max_col, top_col + min_ps + 1)
                feats_list.append(df[df.columns[i]].values)
                feats_info.append((min_ps, top_row, top_col))
                feats_mp[(min_ps, 0, 0, top_row, top_col)] = feats_count
                feats_count = feats_count + 1
            feats_size_list.append(feats_count)

            for now_ps in patch_size_list:
                if now_ps == min_ps:
                    continue
                for start_row in range(-now_ps + min_ps, 1, min_ps):
                    for start_col in range(-now_ps + min_ps, 1, min_ps):
                        if args.using_Swin == 0 and (start_col != 0 or start_row != 0):
                            continue
                        feats_count_now = 0

                        for top_row in range(start_row, max_row, now_ps):
                            for top_col in range(start_col, max_col, now_ps):
                                flag_sub = 0
                                feats_sub_list = []
                                for SR in range(-now_ps // args.magnification_scale + min_ps, 1, min_ps):
                                    for SC in range(-now_ps // args.magnification_scale + min_ps, 1, min_ps):
                                        for TR in range(top_row, top_row + now_ps,now_ps // args.magnification_scale):
                                            for TC in range(top_col, top_col + now_ps,now_ps // args.magnification_scale):
                                                idx = (now_ps // args.magnification_scale, SR, SC, TR, TC)
                                                if  feats_mp.__contains__(idx) \
                                                        and TR + now_ps // args.magnification_scale <= top_row + now_ps \
                                                        and TC + now_ps // args.magnification_scale <= top_col + now_ps:
                                                    flag_sub += 1
                                                    feats_sub_list.append(feats_list[feats_mp[idx]])
                                # print(now_ps,'-',flag_sub)
                                if flag_sub >= 1:
                                    feats_sub_list = np.array(feats_sub_list)
                                    feats_sub_list = torch.from_numpy(feats_sub_list)
                                    feats_sub_list = feats_sub_list.to(torch.float32)
                                    feats_sub=torch.mean(feats_sub_list,0,keepdim=False)
                                    feats_list.append(feats_sub.numpy())
                                    feats_info.append((now_ps, top_row, top_col))
                                    feats_mp[(now_ps, start_row, start_col, top_row, top_col)] = feats_count
                                    feats_count = feats_count + 1
                                    feats_count_now = feats_count_now + 1
                        feats_size_list.append(feats_count_now)



            feats_list = np.array(feats_list)
            feats_list = torch.from_numpy(feats_list)
            feats_list = feats_list.to(torch.float32)
            pe = positionalencoding2d(args.in_classes, 1000, 1000)

            for idx, i in enumerate(feats_list):
                poss=feats_info[idx]
                pxx=(poss[1]+poss[0])//2//256
                pyy=(poss[2]+poss[0])//2//256
                add=pe[:,pxx ,pyy]
                feats_list[idx] += add
            sur_time = np.array(sur_time)
            sur_time = torch.from_numpy(sur_time)
            sur_time = sur_time.to(torch.long)
            censor = np.array(censor)
            censor = torch.from_numpy(censor)
            censor = censor.to(torch.long)
            data_map[WSI_name] = {"feats_list": feats_list, "sur_time": sur_time, "censor": censor,
                                  "feats_size_list": feats_size_list, "feats_info": feats_info}
            # print("building:", time.time() - tim)
        # tim=time.time()
        feats_list = copy.deepcopy(data_map[WSI_name]["feats_list"]).cuda()
        sur_time = copy.deepcopy(data_map[WSI_name]["sur_time"]).cuda()
        censor = copy.deepcopy(data_map[WSI_name]["censor"]).cuda()
        #feats_size_list = copy.deepcopy(data_map[WSI_name]["feats_size_list"])
        #feats_info = copy.deepcopy(data_map[WSI_name]["feats_info"])
        # print("mapping:",time.time()-tim)
        #return feats_list, sur_time, censor, feats_size_list, feats_info
        return feats_list, sur_time, censor
    else :
        if data_map.__contains__(WSI_name) == False:
            df = pandas.read_csv(csv_path)
            feats_list = []
            for i in range(0, df.shape[1]):
                feats_list.append(df[df.columns[i]].values)
            feats_list = np.array(feats_list)
            feats_list = torch.from_numpy(feats_list)
            feats_list = feats_list.to(torch.float32)
            sur_time = np.array(sur_time)
            sur_time = torch.from_numpy(sur_time)
            sur_time = sur_time.to(torch.long)
            censor = np.array(censor)
            censor = torch.from_numpy(censor)
            censor = censor.to(torch.long)
            data_map[WSI_name] = {"feats_list": feats_list, "sur_time": sur_time, "censor": censor}
        # tim
        feats_list = copy.deepcopy(data_map[WSI_name]["feats_list"]).cuda()
        sur_time = copy.deepcopy(data_map[WSI_name]["sur_time"]).cuda()
        censor = copy.deepcopy(data_map[WSI_name]["censor"]).cuda()
        return feats_list, sur_time,censor


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe