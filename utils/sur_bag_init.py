import openslide
import pandas as pd
import glob
import os

from openslide import open_slide


def sur_get_tcga_luad_bags(args):
    sys_name_list=[]
    luad_list = glob.glob(os.path.join('/home/data/liuyong/ms_gcn/TCGA_LUAD_Feats', '*'+str(args.patch_size)+'_0_0.csv'))
    for i in luad_list:
        name = i.split(os.sep)[-1].split('_')[0]
        sys_name_list.append(name)

    df = pd.read_csv('file_us/tcga_luad_all_clean.csv')
    WSI_name_list = []
    sur_time_list = []
    censor_list = []
    bin_sur_time=[0]*30
    for i in range(0, df.shape[0]):
        name=df['slide_id'][i].split('.')[0]
        if name in sys_name_list and name not in WSI_name_list:
            WSI_name_list.append(name)
            sur_time_list.append(int(df['survival_months'][i]/10))
            bin_sur_time[int(df['survival_months'][i]/10)]+=1
            censor_list.append(int(df['censorship'][i]))
    print('bin sur time:',bin_sur_time)
    return WSI_name_list, sur_time_list,censor_list

def sur_get_tcga_lusc_bags(args):
    sys_name_list=[]
    lusc_list = glob.glob(os.path.join('/home/data/liuyong/ms_gcn/TCGA_LUSC_Feats', '*'+str(args.patch_size)+'_0_0.csv'))
    for i in lusc_list:
        name = i.split(os.sep)[-1].split('_')[0]
        sys_name_list.append(name)

    df = pd.read_csv('file_us/tcga_lusc_sur.csv')
    WSI_name_list = []
    sur_time_list = []
    censor_list = []
    bin_sur_time=[0]*30
    for i in range(0, df.shape[0]):
        name=df['slide_id'][i].split('.')[0]
        if name in sys_name_list and name not in WSI_name_list:
            WSI_name_list.append(name)
            sur_time_list.append(int(df['survival_months'][i]/10))
            bin_sur_time[int(df['survival_months'][i]/10)]+=1
            censor_list.append(int(df['censorship'][i]))
    print('bin sur time:',bin_sur_time)
    return WSI_name_list, sur_time_list,censor_list


def sur_get_tcga_ucec_bags(args):
    sys_name_list=[]
    luad_list = glob.glob(os.path.join('/home/data/liuyong/ms_gcn/TCGA_UCEC_Feats', '*'+str(args.patch_size)+'_0_0.csv'))
    for i in luad_list:
        name = i.split(os.sep)[-1].split('_')[0]
        sys_name_list.append(name)

    df = pd.read_csv('file_us/tcga_ucec_all_clean.csv')
    WSI_name_list = []
    sur_time_list = []
    censor_list = []
    bin_sur_time=[0]*30
    for i in range(0, df.shape[0]):
        name=df['slide_id'][i].split('.')[0]
        if name in sys_name_list and name not in WSI_name_list:
            WSI_name_list.append(name)
            sur_time_list.append(int(df['survival_months'][i]/10))
            bin_sur_time[int(df['survival_months'][i]/10)]+=1
            censor_list.append(int(df['censorship'][i]))
    print('bin sur time:',bin_sur_time)
    return WSI_name_list, sur_time_list,censor_list

def sur_get_tcga_brca_idc_bags(args):
    sys_name_list=[]
    brca_list = glob.glob(os.path.join('/home/data/liuyong/ms_gcn/TCGA_BRCA_Feats', '*'+str(args.patch_size)+'_0_0.csv'))
    for i in brca_list:
        name = i.split(os.sep)[-1].split('_')[0]
        sys_name_list.append(name)

    df = pd.read_csv('file_us/tcga_brca_all_clean.csv')
    WSI_name_list = []
    sur_time_list = []
    censor_list = []
    bin_sur_time=[0]*30
    for i in range(0, df.shape[0]):
        name=df['slide_id'][i].split('.')[0]
        oncotree=df['oncotree_code'][i]
        if (name in sys_name_list) and (name not in WSI_name_list) :
            WSI_name_list.append(name)
            sur_time_list.append(int(df['survival_months'][i]/10))
            bin_sur_time[int(df['survival_months'][i]/10)]+=1
            censor_list.append(int(df['censorship'][i]))
    print('bin sur time:',bin_sur_time)
    return WSI_name_list, sur_time_list,censor_list

def sur_get_tcga_blca_bags(args):
    sys_name_list=[]
    list = glob.glob(os.path.join('/home/data/liuyong/ms_gcn/TCGA_BLCA_Feats', '*'+str(args.patch_size)+'_0_0.csv'))
    for i in list:
        name = i.split(os.sep)[-1].split('_')[0]
        sys_name_list.append(name)

    df = pd.read_csv('file_us/tcga_blca_all_clean.csv')
    WSI_name_list = []
    sur_time_list = []
    censor_list = []
    bin_sur_time=[0]*30
    for i in range(0, df.shape[0]):
        name=df['slide_id'][i].split('.')[0]
        if (name in sys_name_list) and (name not in WSI_name_list) :
            WSI_name_list.append(name)
            sur_time_list.append(int(df['survival_months'][i]/10))
            bin_sur_time[int(df['survival_months'][i]/10)]+=1
            censor_list.append(int(df['censorship'][i]))
    print('bin sur time:',bin_sur_time)
    return WSI_name_list, sur_time_list,censor_list

def sur_get_tcga_gbmlgg_bags(args):
    sys_name_list=[]
    list = glob.glob(os.path.join('/home/data/liuyong/ms_gcn/TCGA_GBMLGG_Feats', '*'+str(args.patch_size)+'_0_0.csv'))
    for i in list:
        name = i.split(os.sep)[-1].split('_')[0]
        sys_name_list.append(name)

    df = pd.read_csv('file_us/tcga_gbmlgg_all_clean.csv')
    WSI_name_list = []
    sur_time_list = []
    censor_list = []
    bin_sur_time=[0]*30
    for i in range(0, df.shape[0]):
        name=df['slide_id'][i].split('.')[0]
        if (name in sys_name_list) and (name not in WSI_name_list) :
            WSI_name_list.append(name)
            sur_time_list.append(int(df['survival_months'][i]/10))
            bin_sur_time[int(df['survival_months'][i]/10)]+=1
            censor_list.append(int(df['censorship'][i]))
    print('bin sur time:',bin_sur_time)
    return WSI_name_list, sur_time_list,censor_list






def get_tcga_esca_bags_and_stage_label( label_file_path):
    WSI_name_list = []
    label_list = []
    label_pkl = pd.read_pickle(label_file_path)
    for WSI_name in label_pkl:
        WSI_name_list.append(WSI_name)
        image_stage = label_pkl[WSI_name]
        if image_stage in ['Stage IIB', 'Stage IA', 'Stage IB', 'Stage II', 'Stage I', 'Stage IIA']:
            label_list.append(0)
        else:
            label_list.append(1)
    return WSI_name_list, label_list


