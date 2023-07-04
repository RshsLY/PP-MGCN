import glob
import os
import shutil
import pandas as pd
def f():
    sys_name_list=[]
    lusc_list = glob.glob(os.path.join('/data/liuyong/ms_gcn/TCGA_LUSC_Feats', '*512_0_0.csv'))
    for i in lusc_list:
        name = i.split(os.sep)[-1].split('_')[0]
        sys_name_list.append(name)

    df = pd.read_csv('file_us/tcga_lusc_sur.csv')
    WSI_name_list = []
    sur_time_list = []
    censor_list = []
    bin_sur_time=[0]*25
    for i in range(0, df.shape[0]):
        name=df['slide_id'][i].split('.')[0]
        if name in sys_name_list and name not in WSI_name_list:
            WSI_name_list.append(name)
            sur_time_list.append(int(df['survival_months'][i]/10))
            bin_sur_time[int(df['survival_months'][i]/10)]+=1
            censor_list.append(int(df['censorship'][i]))
    print('bin sur time:',bin_sur_time)
def cul_ass():
    assessment0={'precision': 0.9636363636363636, 'recall': 0.9814814814814815, 'f1': 0.9724770642201834, 'acc': 0.9712918660287081, 'auc': 0.992940960762743}
    assessment1= {'precision': 0.8585858585858586, 'recall': 0.9659090909090909, 'f1': 0.9090909090909091, 'acc': 0.9186602870813397, 'auc': 0.9861006761833208}
    assessment2={'precision': 0.96, 'recall': 0.96, 'f1': 0.96, 'acc': 0.9615384615384616, 'auc': 0.9752777777777778}
    assessment3= {'precision': 0.9224137931034483, 'recall': 0.9553571428571429, 'f1': 0.9385964912280702, 'acc': 0.9326923076923077, 'auc': 0.9876302083333334}
    assessment4={'precision': 0.9696969696969697, 'recall': 0.9230769230769231, 'f1': 0.9458128078817735, 'acc': 0.9471153846153846, 'auc': 0.9860392011834319}
    acc=assessment0['acc']+assessment1['acc']+assessment2['acc']+assessment3['acc']+assessment4['acc']
    acc=acc/5
    print(acc)
    auc=assessment0['auc']+assessment1['auc']+assessment2['auc']+assessment3['auc']+assessment4['auc']
    auc=auc/5
    print(auc)
    f1=assessment0['f1']+assessment1['f1']+assessment2['f1']+assessment3['f1']+assessment4['f1']
    f1=f1/5
    print(f1)
def test_sort():
    s_list=[]
    s_list.append((2,1))
    s_list.append((1,1))
    s_list.append((1,0))
    s_list.append((2,2))
    s_list.append((1,-1))
    s_list.append((2,3))
    s_list.append((2,-1))

    s_list = sorted(s_list, reverse=False)
    print(s_list)
def del_download_dataset():
    svs_list=glob.glob(os.path.join('/data/liuyong/TCGA_BRCA/*/','*.svs'))
    for i in svs_list:
        shutil.move(i,'/data/liuyong/TCGA_BRCA/')
        shutil.rmtree('/data/liuyong/TCGA_BRCA/'+'/'+i.split(os.sep)[-2])
    svs_list = glob.glob(os.path.join('/data/liuyong/TCGA_BRCA/', '*.svs'))
    for i in svs_list:
        name = '/data/liuyong/TCGA_BRCA/' + i.split(os.sep)[-1].split('.')[0] + '.svs'
        os.rename(i, name)
    print(svs_list)
def rm_not_in_csv():
    df=pd.read_csv("/data/liuyong/ms_gcn/file_us/tcga_brca_all_clean.csv")
    svs_list = glob.glob(os.path.join('/data/liuyong/TCGA_BRCA/', '*svs'))
    for i in svs_list:
        flag=0
        for j in range(df.shape[0]):
            name1= df['slide_id'][j].split('.')[0]
            name2=i.split(os.sep)[-1].split(".")[0]
            if name2==name1:
                flag=1
        if flag==0:
            os.remove(i)

rm_not_in_csv()