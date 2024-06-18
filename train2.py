import os
import time
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityRanged, NormalizeIntensityd, AsChannelFirstd,
    RandFlipd, RandAffined, AddChanneld, Spacingd, CropForegroundd,ScaleIntensityd,
    SpatialCropd, RandSpatialCropd, ToTensord)
from glob import glob
import matplotlib.pyplot as plt
from monai.transforms.utility.dictionary import AddChanneld
from monai.inferers import sliding_window_inference
import numpy as np
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.utils import first, set_determinism
from tensorboardX import SummaryWriter
from monai.handlers.utils import from_engine
import SimpleITK as sitk
from init2 import Options
import torch.nn as nn
import torch
from utils import ssim_3d, psnr
from network import RCNets, update_learning_rate
from UNet3D import UNet3D
from loss import PerceptualLoss3D, GeneratorLoss
from DRRN import DRRN
from RDNetwork import RDN



def main():
    opt = Options().parse()
    # monai.config.print_config()

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
    images = sorted(glob(os.path.join(opt.images_folder, '*.nii.gz')))
    labels = sorted(glob(os.path.join(opt.labels_folder, '*.nii.gz')))

    data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(images, labels)
    ]
    train_dicts = data_dicts[0:120]
    

    # Transforms list
    train_transforms = Compose(
        [   
            LoadImaged(keys=["image", "label"]),     # user can also add other random transforms
            AsChannelFirstd(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            #NormalizeIntensityd(keys=["image", "label"], nonzero=False),
            ScaleIntensityd(keys=["image", "label"], minv=0.0, maxv=1.0, factor=None),
            
            CropForegroundd(keys=['image', 'label'], source_key='label', select_fn = lambda x : x > 0, margin=5),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
            RandSpatialCropd(keys=['image', 'label'], random_size = False, roi_size = opt.patch_size),
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('nearest', 'nearest'),
            #     prob=0.5,
            #     rotate_range=(np.pi/90, 0, 0),
            #     scale_range=(0.1, 0.1, 0.1)),
            # ToTensord(keys=['image', 'label']),
        ]
    )

    val_transforms = Compose(
        [   
            LoadImaged(keys=["image", "label"]),     # user can also add other random transforms
            AsChannelFirstd(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            #NormalizeIntensityd(keys=["image", "label"], nonzero=False),
            ScaleIntensityd(keys=["image", "label"], minv=0.0, maxv=1.0, factor=None),
            # ScaleIntensityRanged(
            #     keys=["image", "label"], a_min=0, a_max=100,
            #     b_min=0.0, b_max=1.0, clip=True,
            # ),
            #SpatialCropd(keys=['image', 'label'], roi_center=[15, 256, 256], roi_size=[32, 384, 384]),
            CropForegroundd(keys=['image', 'label'], source_key='label', select_fn = lambda x : x > 0, margin=5),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
            RandSpatialCropd(keys=['image', 'label'], random_size = False, roi_size = opt.patch_size),
            
            ToTensord(keys=['image', 'label']),
        ]
    )   

    # create a training data loader
    check_train = Dataset(data=train_dicts, transform=train_transforms)
    train_loader = DataLoader(check_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=torch.cuda.is_available())

    check_val = Dataset(data=val_dicts, transform=val_transforms)
    val_loader = DataLoader(check_val, batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=torch.cuda.is_available())

    #net = UNet3D(in_channels=1, out_channels=1, init_features=32)
    net = RDN(scale_factor=4,
                num_channels=1,
                num_features=64,
                growth_rate=64,
                num_blocks=6,
                num_layers=5)
    net.cuda()
    if num_gpus > 1:
        net = nn.DataParallel(net)
    
    if opt.preload is not None:
        net.load_state_dict(torch.load(opt.preload))

    
    loss_function = nn.L1Loss().cuda()
    
    optimizer = torch.optim.Adam(net.parameters(), opt.lr)
    net_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / opt.epochs) ** 0.9)
    best_metric = -1
    best_metric_epoch = -1
    
    
    for epoch in range(opt.epochs):
        print("--" * 30)
        print(f"epoch {epoch + 1}/{opt.epochs}")
        net.train()
        
        running_total_loss = 0.
        running_d_loss = 0.
        running_ssim = 0.
        running_psnr = 0.
        step = 0
        for batch_data in train_loader:
            now = time.time()
            step += 1
            inputs, labels = batch_data["image"].cuda(), batch_data["label"].cuda()
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            # loss = loss.cuda()
            loss.backward()
            optimizer.step()

            ssim_value = ssim_3d(outputs, labels)
            psnr_value = psnr(outputs, labels)

            running_total_loss += loss.detach().item()
            running_ssim += ssim_value.item()
            #running_ms_ssim += ms_ssim_value.item()
            running_psnr += psnr_value.item()

            consume_time = time.time() - now
            now = time.time()
            if step % 1 == 0:
                print("epoch:{},  Step {}/{}, loss:{:.4f},  ssim:{:.5f},  psnr:{:.5f}, spent time:{:.2f}s".format(
                epoch + 1, step, len(train_loader), loss.detach().item(), ssim_value.item(), psnr_value.item(),
                consume_time))
        print('epoch:{},  total_loss:{:.4f},   SSIM:{:.5f},  PSNR:{:.5f}.'.format(epoch + 1,
        running_total_loss / step,  running_ssim / step, running_psnr / step))
       

        writer.add_scalar('loss', loss.detach().item()/step, epoch + 1)

        
        if (epoch + 1) % opt.save_model_epochs == 0 or (epoch + 1) == opt.epochs:
            torch.save(net.state_dict(), f'{save_model_path}/RDN3D_Flair_{time_str}_last_{epoch + 1}.pth')

if __name__ == "__main__":
    main()