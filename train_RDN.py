import os
import time

from glob import glob
import matplotlib.pyplot as plt

import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import SimpleITK as sitk
from init import Options
import torch.nn as nn
import torch
from utils import ssim_3d, psnr
from Model import RDN
from Brain_data import MRIDataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_msssim import ssim
from utils import psnr



def main():
    opt = Options().parse()

    # check gpus
    if opt.gpu_ids != '-1':
        num_gpus = len(opt.gpu_ids.split(','))
    else:
        num_gpus = 0
    print('number of GPU:', num_gpus)

    
    # create checkpoints
    if not os.path.exists(opt.save_dir):
        os.makedirs(f'{opt.save_dir}/')
    
    time_str=time.strftime("%y%m%d_%H%M", time.localtime())
    os.makedirs(f'{opt.save_dir}/{time_str}') 
    save_model_path = f'{opt.save_dir}/{time_str}'

    writer = SummaryWriter()

    # train images
    lr_files = sorted([f for f in os.listdir(opt.lr_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    hr_files = sorted([f for f in os.listdir(opt.hr_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
 
    net = RDN(num_channels=1,
              num_features=64,
              growth_rate=64,
              num_blocks=16,
              num_layers=6)
    
    net.cuda()
    if num_gpus > 1:
        net = nn.DataParallel(net)
    
    if opt.preload is not None:
        net.load_state_dict(torch.load(opt.preload))

    
    loss_function = nn.L1Loss().cuda()
    
    optimizer = torch.optim.Adam(net.parameters(), opt.lr)
    
    
    
    for epoch in range(opt.epochs):
        print("--" * 30)
        print(f"epoch {epoch + 1}/{opt.epochs}")
        
        net.train()

        for f_i in range(len(lr_files)):
            lr_path = os.path.join(opt.lr_dir,lr_files[f_i])
            hr_path = os.path.join(opt.lr_dir,hr_files[f_i])
            print("--" * 10,lr_path)
            coronal_dataset = MRIDataset(lr_path, hr_path, plane='coronal')
            sagittal_dataset = MRIDataset(lr_path, hr_path, plane='sagittal')
            coronal_loader = DataLoader(coronal_dataset, batch_size=opt.batch_size, shuffle=True)
            sagittal_loader = DataLoader(sagittal_dataset, batch_size=opt.batch_size, shuffle=True)
        
            running_loss = 0.
            running_ssim = 0.
            running_psnr = 0.
            step = 1
            for lr_coronal, hr_coronal in coronal_loader:
                now = time.time()
                lr, hr = lr_coronal.cuda(), hr_coronal.cuda()
                
                optimizer.zero_grad()
                outputs = net(lr)
                loss1 = loss_function(outputs, hr)
                # loss = loss.cuda()
                loss1.backward()
                optimizer.step()
                
                ssim_value = ssim(outputs, hr, data_range=1, size_average=True)
                psnr_value = psnr(outputs, hr)

                running_loss += loss1.detach().item()
                running_ssim += ssim_value.item()
                #running_ms_ssim += ms_ssim_value.item()
                running_psnr += psnr_value.item()

                consume_time = time.time() - now
                now = time.time()
                step += 1
                if step % 1 == 0:
                    print("epoch:{},  Step {}/{}, loss:{:.4f},  ssim:{:.5f},  psnr:{:.5f}, spent time:{:.2f}s".format(
                    epoch + 1, step, len(coronal_loader), loss1.detach().item(), ssim_value.item(), psnr_value.item(),
                    consume_time))
            print('epoch:{},  total_loss:{:.4f},   SSIM:{:.5f},  PSNR:{:.5f}.'.format(epoch + 1,
            running_loss / len(coronal_loader),  running_ssim / len(coronal_loader), running_psnr / len(coronal_loader)))
        

            writer.add_scalar('loss', loss1.detach().item()/len(coronal_loader), epoch + 1)

            running_loss = 0.
            running_ssim = 0.
            running_psnr = 0.
            step = 1

            for lr_sagittal, hr_sagittal in sagittal_loader:
                
                now = time.time()
                lr2, hr2 = lr_sagittal.cuda(), hr_sagittal.cuda()
                
                optimizer.zero_grad()
                outputs2 = net(lr2)
                loss2 = loss_function(outputs2, hr2)
                # loss = loss.cuda()
                loss2.backward()
                optimizer.step()

                ssim_value2 = ssim(outputs2, hr2, data_range=1, size_average=True)
                psnr_value2 = psnr(outputs2, hr2)

                running_loss += loss2.detach().item()
                running_ssim += ssim_value2.item()
                #running_ms_ssim += ms_ssim_value.item()
                running_psnr += psnr_value2.item()

                consume_time = time.time() - now
                now = time.time()
                step += 1
                if step % 1 == 0:
                    print("epoch:{},  Step {}/{}, loss:{:.4f},  ssim:{:.5f},  psnr:{:.5f}, spent time:{:.2f}s".format(
                    epoch + 1, step, len(sagittal_loader), loss2.detach().item(), ssim_value2.item(), psnr_value2.item(),
                    consume_time))
            print('epoch:{},  total_loss:{:.4f},   SSIM:{:.5f},  PSNR:{:.5f}.'.format(epoch + 1,
            running_loss / len(sagittal_loader),  running_ssim / len(sagittal_loader), running_psnr / len(sagittal_loader)))
        

            writer.add_scalar('loss2', loss2.detach().item()/len(sagittal_loader), epoch + 1)

            
            if (epoch + 1) % opt.save_model_epochs == 0 or (epoch + 1) == opt.epochs:
                torch.save(net.state_dict(), f'{save_model_path}/RDN_Flair_{time_str}_last_{epoch + 1}.pth')

if __name__ == "__main__":
    main()