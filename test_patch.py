from utils import *
import numpy as np
import argparse
import os

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import SimpleITK as sitk
# from UNet3D import UNet3D
from monai.transforms import (Compose, ScaleIntensity, CropForegroundd,apply_transform)
from RDNetwork import RDN
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--train_ls', type=str, default='/home/hydu/MRI_SR/processMRI2/FLAIR/data/LR_BSpline/')
parser.add_argument('--train_hs', type=str, default='/home/hydu/MRI_SR/processMRI2/FLAIR/data/SR_BSpline/')
parser.add_argument("--results", type=str, default='./result', help='path to the .nii result to save')
parser.add_argument("--weights", type=str, default='./checkpoints/231023_1610/RDN3D_Flair_231023_1610_last_440.pth', help='network weights to load')
parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
args = parser.parse_args()

def test():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    
    time_str=time.strftime("%y%m%d_%H%M", time.localtime())
    name = args.weights.split('/')[-1][:-4]
    
    save_input_path = f'{args.results}/{name}/LS_nii/'
    save_pred_path = f'{args.results}/{name}/pred_nii/'
    save_gt_path = f'{args.results}/{name}/HS_nii/'
    
    if not os.path.exists(save_input_path):
        os.makedirs(save_input_path)
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)
    if not os.path.exists(save_gt_path):
        os.makedirs(save_gt_path)
    print("--" * 30,'load data')


    model = RDN(scale_factor=4,
                    num_channels=1,
                    num_features=64,
                    growth_rate=64,
                    num_blocks=6,
                    num_layers=4).cuda()
    model.eval()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.weights))

    running_ssim = 0.
    running_ms_ssim = 0.
    running_psnr = 0.
    step = 120
    kkk = 0

    ls_dirlist, hs_dirlist, listname = listpath(args.train_ls,args.train_hs)
    #print(listname)
    now = time.time()
    trans_transform = Compose([ScaleIntensity(minv=0.0, maxv=1.0, factor=None)])
    for i in range(step,len(ls_dirlist)):
        img = ants.image_read(ls_dirlist[i])
        labels = ants.image_read(hs_dirlist[i])
        
        image_array = img.numpy()
        image_array = apply_transform(trans_transform,image_array)

        labels_array = labels.numpy()
        labels_array = apply_transform(trans_transform,labels_array)
        reimg = ants.from_numpy(image_array, origin=img.origin, spacing=img.spacing,
                        direction=img.direction, has_components=False)
        relabels = ants.from_numpy(labels_array, origin=labels.origin, spacing=labels.spacing,
                        direction=labels.direction, has_components=False)

        img_patches = extract_image_patches(reimg, patch_size=(64, 64, 64), stride_length=31)
        
        predlist = []
        for j, test_input in enumerate(img_patches):
            test_input = test_input.transpose(2,0,1)
            test_input = test_input[np.newaxis, np.newaxis, :, :, :]
            test_input = torch.from_numpy(test_input).float()
            test_input = test_input.cuda()
            with torch.no_grad():
                test_pred = model(test_input)
                test_pred = test_pred[0,0].cpu().numpy()
                test_pred = test_pred.transpose(1,2,0)
            predlist.append(test_pred)
        pred_image = reconstruct_image_from_patches(predlist, img, stride_length=31)
        
        ssim_value, ms_ssim_value= ssim_3d_all(pred_image, relabels)
        psnr_value = psnr_np(pred_image, relabels)

        running_ssim += ssim_value.item()
        running_ms_ssim += ms_ssim_value.item()
        running_psnr += psnr_value.item()

        consume_time = time.time() - now
        now = time.time()

        print("Case {},  ssim:{:.5f}, ms_ssim:{:.5f}, psnr:{:.5f}, spent time:{:.2f}s".format(
            listname[i], ssim_value.item(), ms_ssim_value.item(), psnr_value.item(), consume_time))
        
        saved_input_name = save_input_path + listname[i] + '_ls.nii.gz'
        saved_pred_name = save_pred_path + listname[i] + '_pred.nii.gz'
        saved_gt_name = save_gt_path + listname[i] + '_hs.nii.gz'

        ants.image_write(reimg, saved_input_name)
        ants.image_write(pred_image, saved_pred_name)
        ants.image_write(relabels, saved_gt_name)

    print('SSIM:{:.5f}, MS_SSIM:{:.5f},  PSNR:{:.5f}.'.format(running_ssim / (i + 1), running_ms_ssim / (i + 1), running_psnr / (i + 1)))


if __name__ == "__main__":
    test()
