import pandas as pd
import argparse
import glob
from openslide import open_slide
import sys
from datetime import  datetime
from PIL import Image, ImageFilter, ImageStat
import torch.nn as nn
import torch
import numpy as np
import torchvision
from torchvision import  transforms
import os
from torch.utils.data import Dataset, DataLoader
import openslide
import  cv2

class My_dataloader(Dataset):

    def __init__(self, patch_list, pos_list,transform):
        self.patch_list = patch_list
        self.pos_list=pos_list
        self.transform = transform

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        img = self.patch_list[idx]
        pos = self.pos_list[idx]
        if self.transform:
            img = self.transform(img)
        return img,pos


def patch_extract_and_compute_feature(patch_size,slide_path,args,model,bag_idx,num_bags,background_threshold):
    slide = open_slide(slide_path)
    patch_size_list = list(args.patch_size_list)
    base_power = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if base_power!='40' and base_power!='20' :
        print("base_power not in [40,20]!")
        return
    if base_power=='40':
        print("  40x,need resize")
        patch_size=patch_size*2
        for i,j in enumerate(patch_size_list):
            patch_size_list[i]=j*2
    min_size = patch_size_list[0]
    slide_height = slide.dimensions[1]
    slide_width = slide.dimensions[0]
    slide_window_count=1
    if args.slide_window_using==True:
        slide_window_count=patch_size//min_size
    for window_begin_row_idx in range(0,slide_window_count):
        for window_begin_column_idx in range (0,slide_window_count):
            start_row_pixel=window_begin_row_idx*min_size
            start_column_pixel=window_begin_column_idx*min_size
            if start_row_pixel>0:
                start_row_pixel=start_row_pixel-patch_size
            if start_column_pixel>0:
                start_column_pixel=start_column_pixel-patch_size

            if base_power == '40':
                feats_path = (os.path.join(args.feature_path))
                slide_name = slide_path.split(os.path.sep)[-1].split('.')[0]
                csv_pp=os.path.join(feats_path, slide_name + '_' + str(patch_size // 2) + '_' + str(
                    start_row_pixel // 2) + '_' + str(start_column_pixel // 2) + '.csv')
                if os.path.exists(csv_pp):
                    print(" exist csv",csv_pp)
                    continue
            if base_power == '20':
                feats_path = (os.path.join(args.feature_path))
                slide_name = slide_path.split(os.path.sep)[-1].split('.')[0]
                csv_pp=os.path.join(feats_path, slide_name + '_' + str(patch_size) + '_' + str(
                    start_row_pixel) + '_' + str(start_column_pixel ) + '.csv')
                if os.path.exists(csv_pp):
                    print(" exist csv",csv_pp)
                    continue

            patch_list = []
            pos_list = []
            patch_count = 0
            for top_row in range(start_row_pixel,slide_height,patch_size):
                for top_colum in range(start_column_pixel,slide_width,patch_size):
                    patch = slide.read_region((top_colum,top_row), 0, (patch_size, patch_size))
                    patch = patch.convert('RGB')
                    if base_power == '40':
                        patch=patch.resize((patch_size//2,patch_size//2))
                    if args.is_using_background_threshold == False:
                        patch_list.append(patch)
                        pos=""
                        if base_power=='40':
                            pos=str(top_row//2)+','+str(top_colum//2)
                        if base_power == '20':
                            pos=str(top_row) + ',' + str(top_colum)
                        pos_list.append(pos)
                        patch_count = patch_count + 1
                        if patch_count % 10 == 0:
                            sys.stdout.write('\r  patch size:%d    now_top:(%s)    patch:[%d/%d]' % (patch_size//2, pos, patch_count,slide_width*slide_height//patch_size//patch_size))
                    if args.is_using_background_threshold == True:
                        edge = patch.filter(ImageFilter.FIND_EDGES)
                        edge = ImageStat.Stat(edge).sum
                        if base_power == '40':
                            edge = np.mean(edge) / ((patch_size//2) ** 2)
                        if base_power == '20':
                            edge = np.mean(edge) / ((patch_size) ** 2)
                        if edge > background_threshold:
                            patch_list.append(patch)
                            pos = ""
                            if base_power == '40':
                                pos = str(top_row // 2) + ',' + str(top_colum // 2)
                            if base_power == '20':
                                pos = str(top_row) + ',' + str(top_colum)
                            pos_list.append(pos)
                            #patch.save(os.path.join('wsi',str(patch_size//2)+pos+'.jpeg'))
                            patch_count=patch_count+1
                            if patch_count%10 ==0 :
                                sys.stdout.write('\r  patch size:%d    now_top:(%s)    patch:[%d/%d]' % (patch_size//2, pos, patch_count, slide_width * slide_height // patch_size // patch_size))
            # once
            trans = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            imagedataset = My_dataloader(patch_list, pos_list, trans)
            dataloader = torch.utils.data.DataLoader(imagedataset, batch_size=args.batch_size, shuffle=False,
                                                     drop_last=False)
            model.eval()
            feats_list = []
            pos_list_now = []
            with torch.no_grad():
                for ii, (imgs, poss) in enumerate(dataloader):
                    imgs = imgs.cuda()
                    feats, _ = model(imgs)
                    feats = feats.cpu().numpy()
                    feats_list.extend(feats)
                    pos_list_now.extend(poss)
                    sys.stdout.write('\r  Computed: {}/{} -- {}/{}'.format(bag_idx+1, num_bags, ii + 1, len(dataloader)))
            dict_save = {}
            for (i, j) in zip(feats_list, pos_list_now):
                dict_save[j] = i
            if len(feats_list) == 0:
                print('No valid patch extracted from: ' + slide_path)
            else:
                df = pd.DataFrame(dict_save)
                feats_path = (os.path.join(args.feature_path))
                os.makedirs(feats_path, exist_ok=True)
                slide_name = slide_path.split(os.path.sep)[-1].split('.')[0]
                if base_power == '40':
                    df.to_csv(os.path.join(feats_path,slide_name + '_' + str(patch_size//2) + '_' + str(start_row_pixel//2) + '_' + str(start_column_pixel//2) + '.csv'),
                              index=False, float_format='%.4f')
                if base_power == '20':
                    df.to_csv(os.path.join(feats_path,slide_name + '_' + str(patch_size) + '_' + str(start_row_pixel) + '_' + str(start_column_pixel) + '.csv'),
                              index=False, float_format='%.4f')


class fully_connected(nn.Module):
	def __init__(self, model, num_ftrs, num_classes):
		super(fully_connected, self).__init__()
		self.model = model
		self.fc_4 = nn.Linear(num_ftrs,num_classes)

	def forward(self, x):
		x = self.model(x)
		x = torch.flatten(x, 1)
		out_1 = x
		out_3 = self.fc_4(x)
		return  out_1, out_3

def load_KimiaNet(args):

    model = torchvision.models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    num_ftrs = model.classifier.in_features
    model_final = fully_connected(model.features, num_ftrs,30)
    model_final = model_final.cuda()
    model_final = nn.DataParallel(model_final)
    model_final.load_state_dict(torch.load('file_us/KimiaNetPyTorchWeights.pth'))
    return model_final

if __name__ == '__main__':
    #防止图片过大会报错
    Image.MAX_IMAGE_PIXELS = None
    parser = argparse.ArgumentParser(description='Patch extraction for WSI')
    parser.add_argument('--dataset_path', type=str, default='/data/liuyong/TCGA_GBMLGG', help='Dataset path')
    parser.add_argument('--feature_path', type=str, default='TCGA_GBMLGG_Feats', help='path for extracted feature')
    parser.add_argument('--patch_format', type=str, default='.jpeg', help='Image format for patches')
    parser.add_argument('--slide_format', type=str, default='.svs', help='Image format for patches')
    parser.add_argument('--patch_size_list', type=int, nargs='+', default=(512,1024), help='patch size list')
    parser.add_argument('--slide_window_using', type=bool, default=False, help='is using sliding window')
    parser.add_argument('--is_using_background_threshold', type=bool, default=True, help='Is using Threshold for filtering background ')
    parser.add_argument('--background_threshold', type=int, nargs='+',default=(8,4), help='Threshold for filtering background 512-8 1024-4 2048-2')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s)')
    parser.add_argument('--batch_size', type=int,default=1, help='batch_size for network')

    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
    dataset_path = os.path.join(args.dataset_path)
    all_slides = glob.glob(os.path.join(dataset_path, '*' + args.slide_format)) \
                 + glob.glob(os.path.join(dataset_path, '*/*' + args.slide_format)) \
                 + glob.glob(os.path.join(dataset_path, '*/*/*' + args.slide_format))
    patch_size_listt = list(args.patch_size_list)
    background_threshold_list = list(args.background_threshold)
    for i in range(0,patch_size_listt.__len__()-1):
        assert patch_size_listt[i+1]%patch_size_listt[i]==0
        assert patch_size_listt[i+1]//patch_size_listt[i]==2
    print("patch_size:",patch_size_listt,"  bg_th:",background_threshold_list,"  gpu:",gpu_ids)
    model=load_KimiaNet(args)
    for patch_size_idx,patch_size in enumerate(patch_size_listt):
        for idx, slide_path in enumerate(all_slides):
            print('\nProcess patch_size:{} slide:{}/{} slide_name:{}  {}'.format(patch_size,idx + 1, len(all_slides),slide_path.split()[-1],datetime.now()))
            patch_extract_and_compute_feature(patch_size,slide_path,args,model,idx,all_slides.__len__(),background_threshold_list[patch_size_idx])
    print('\nPatch extraction done for {} slides.'.format(len(all_slides)))
