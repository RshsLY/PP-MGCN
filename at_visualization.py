import PIL.Image
import numpy as np
import openslide

import ahis_file.sur_MS_GCN as mil
import  argparse
import  datetime
import os
import torch
import utils.sur_bag_build
import cv2

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str,
                        default="saved_model/TCGA_BLCA_survival_sur_MS_GCN_2_2022-09-25-02-24-06.415848.pth",
                        help="path for saved model")
parser.add_argument("--feats_path",type=str,default="TCGA_BLCA_Feats/TCGA-5N-A9KM-01Z-00-DX1_512_0_0.csv")
parser.add_argument("--tif_path",type=str,default="/data/liuyong/TCGA_BLCA/TCGA-5N-A9KM-01Z-00-DX1.svs")
parser.add_argument("--WSI_name",type=str,default="TCGA-5N-A9KM-01Z-00-DX1")
parser.add_argument("--layer_select", type=int,default=0)
parser.add_argument("--down", type=int,default=64)


parser.add_argument("--patch_size", type=int,            default=512,               help="patch_size to use")
parser.add_argument('--gpu_index', type=int,             default=0,                 help='GPU ID(s)')
parser.add_argument("--dataset", type=str,               default="TCGA_BLCA",       help="Database to use[TCGA_LUAD,TCGA_LUSC,TCGA_UCEC,TCGA_BLCA,TCGA_BLCA,TCGA_BLCA]")
parser.add_argument("--model", type=str,                 default="sur_MS_GCN",      help="Model to use[sur_MIL_mean,sur_MIL_max,sur_ABMIL,sur_Patch_GCN,sur_DSMIL,sur_TransMIL,sur_H2_MIL,sur_MS_GCN]")
parser.add_argument("--in_classes", type=int,            default=1024,              help="Feature size of each patch")
parser.add_argument("--out_classes", type=int,           default=30,                help="Number of classes,UCEC,LUAD=25 BLCA=30")
#------MS_GCN
parser.add_argument("--ms_gcn_layer", type=int,          default=3,                 help="[1,4]")
parser.add_argument("--ms_gcn_is_slide",type=int,        default=1,                 help="[0,1]")
parser.add_argument("--gcn_layer", type=int,             default=3)
#-----MS_GCN
parser.add_argument("--model_save_path", type=str,       default="saved_model",     help="path for save model")
parser.add_argument("--task", type=str,                  default="survival",        help="Task of classification[survival]")
parser.add_argument("--divide_seed", type=int,           default=0,              help="Data division seed")
# ------------------
parser.add_argument("--batch_size", type=int,            default=32,                help="Data volume of model training once")
parser.add_argument("--epochs", type=int,                default=200,               help="Cycle times of model training")
parser.add_argument("--epochs_patience", type=int,       default=32,                help="epoch num of no update to stop")
parser.add_argument("--epochs_warm", type=int,           default=8,                 help="epoch num for warm up")
parser.add_argument("--drop_out_ratio", type=float,      default=0.25,               help="Drop_out_ratio")
parser.add_argument("--lr", type=float,                  default=0.00001,           help="Learning rate of model training")
parser.add_argument("--weight_decay", type=float,        default=0.000001,          help="weight_decay of model training")
# ------------------
parser.add_argument("--number_kfold", type=int,          default=5,                 help="Number of KFold")
parser.add_argument("--number_run", type=int,            default=1,                 help="Number of runs")
parser.add_argument("--time_stamp",type=str,             default="-1")
args, _ = parser.parse_known_args()
args.time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')
gpu_ids = tuple((args.gpu_index,))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)


tif=openslide.OpenSlide(args.tif_path)
print(tif.level_dimensions[0][0],tif.level_dimensions[0][1])
model = mil.MIL(args)
model = model.cuda()
model.load_state_dict(torch.load(args.model_path))
model.eval()


pil_img=tif.get_thumbnail((tif.level_dimensions[0][0]//args.down,tif.level_dimensions[0][1]//args.down))
pil_img.show()

feats_list, sur_time, censor, edge_index, edge_index_diff, feats_size_list, feats_info = \
    utils.sur_bag_build.get_bag(args, args.WSI_name, -1, -1, {})
prediction_list,at_=model(feats_list,edge_index,edge_index_diff,feats_size_list)

at_c=(at_[0]-torch.min(at_[0][0]))*255.0/torch.max(at_[0][0])
for i in range(1,3):
    at_[i]=(at_[i]-torch.min(at_[i][0]))*255.0/torch.max(at_[i][0])
    at_c=torch.cat((at_c,at_[i]),dim=-1)

at_c=at_c.cpu().detach().numpy()

heat_map=np.zeros([tif.level_dimensions[0][1]//args.down,tif.level_dimensions[0][0]//args.down])
is_color=np.zeros([tif.level_dimensions[0][1]//args.down,tif.level_dimensions[0][0]//args.down])

for idx,(ps,tr,tc) in enumerate(feats_info):
    now_ps=args.patch_size
    beishu=1
    for i in range(args.layer_select) :
        now_ps=now_ps*2
        beishu*=4
    if ps != now_ps: continue
    ps1=ps//args.down
    tr1=tr//args.down
    tc1=tc//args.down
    if tif.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)=='40':
        ps1*=2
        tr1*=2
        tc1*=2
    for i in range(tr1,ps1+tr1):
        for j in range(tc1,ps1+tc1):
            if i < tif.level_dimensions[0][1]//args.down and j <tif.level_dimensions[0][0]//args.down:
                heat_map[i][j]+=  at_c[0][idx]/beishu
                is_color[i][j]=1
heat_map=heat_map.astype(np.uint8)
heat_map1=cv2.applyColorMap(heat_map,cv2.COLORMAP_JET) #BGR
heat_map1=cv2.cvtColor(heat_map1,cv2.COLOR_BGR2RGB)
for i in range(len(is_color)):
    for j in range(len(is_color[0])):
        if is_color[i][j]==0:
            heat_map1[i][j][0] = 255
            heat_map1[i][j][1] = 255
            heat_map1[i][j][2] = 255
heat_map1=heat_map1.astype(np.uint8)
heat_map1=PIL.Image.fromarray(heat_map1)
heat_map1.show()
final_img=PIL.Image.blend(pil_img,heat_map1,0.5)
final_img.show()


pil_img.save("ori"+args.WSI_name+str(args.layer_select)+'.png')
heat_map1.save("heatmap"+args.WSI_name+str(args.layer_select)+'.png')
final_img.save("final"+args.WSI_name+str(args.layer_select)+'.png')