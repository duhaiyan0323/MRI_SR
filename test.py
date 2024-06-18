import os
import time
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityRanged, Rotate90d,
    AddChanneld, Spacingd, CropForegroundd, AsChannelFirstd, ScaleIntensityd,
    SpatialCropd, RandSpatialCropd, ToTensord)
from glob import glob
import matplotlib.pyplot as plt
from monai.transforms.utility.dictionary import AddChanneld
import numpy as np
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.utils import first, set_determinism
import argparse
from tensorboardX import SummaryWriter
from monai.handlers.utils import from_engine
import SimpleITK as sitk
from init import Options
import torch.nn as nn
import torch
from utils import ssim_3d, psnr, saved_preprocessed
from network import RCNets
from UNet3D import UNet3D
from loss import PerceptualLoss3D, GeneratorLoss
from RDNetwork import RDN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--images_folder', type=str, default='/home/hydu/MRI_SR/processMRI2/FLAIR/data/LR_BSpline/')
parser.add_argument('--labels_folder', type=str, default='/home/hydu/MRI_SR/processMRI2/FLAIR/data/SR_BSpline/')
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
    images = sorted(glob(os.path.join(args.images_folder, '*.nii.gz')))
    labels = sorted(glob(os.path.join(args.labels_folder, '*.nii.gz')))
    list_labels = sorted(glob(os.path.join(args.labels_folder, '*.nii.gz')))

    data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(images, labels)
    ]

    val_dicts = data_dicts[120:150]


    test_transforms = Compose(
        [   
            LoadImaged(keys=["image", "label"]),     # user can also add other random transforms
            AsChannelFirstd(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            ScaleIntensityd(keys=["image", "label"], minv=0.0, maxv=1.0, factor=None),
            # ScaleIntensityRanged(
            #     keys=["image", "label"], a_min=0, a_max=100,
            #     b_min=0.0, b_max=1.0, clip=True,
            # ),
            #SpatialCropd(keys=['image', 'label'], roi_center=[15, 256, 256], roi_size=[32, 384, 384]),
            CropForegroundd(keys=['image', 'label'], source_key='label', select_fn = lambda x : x > 0, margin=0, k_divisible = 16),
            Rotate90d(keys=['image', 'label'],k=3, spatial_axes=(1, 2)),
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('nearest', 'nearest'),
            #     prob=0.5,
            #     rotate_range=(np.pi/90, 0, 0),
            #     scale_range=(0.1, 0.1, 0.1)),
            ToTensord(keys=['image', 'label']),
        ]
    )

    # create a training data loader
    check_train = Dataset(data=val_dicts, transform=test_transforms)
    test_loader = DataLoader(check_train, batch_size=1, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())
    print(len(test_loader))
    print('-----load finish-------')
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
    running_psnr = 0.
    step = 120
    kkk = 0
    for batch_data in test_loader:
        now = time.time()
        test_inputs, test_labels = batch_data["image"].cuda(), batch_data["label"].cuda()
        print(test_inputs.size())
        with torch.no_grad():
            test_pred = model(test_inputs)
        
        ssim_value = ssim_3d(test_pred, test_labels)
        psnr_value = psnr(test_pred, test_labels)

        running_ssim += ssim_value.item()
        running_psnr += psnr_value.item()

        consume_time = time.time() - now
        now = time.time()

        print("Case {},  size:{}, ssim:{:.5f},  psnr:{:.5f}, spent time:{:.2f}s".format(
            list_labels[step].split('/')[-1][0:7], test_inputs.size(), ssim_value.item(), psnr_value.item(), consume_time))

        
        sitk_imgs = sitk.ReadImage(list_labels[step])
        input = test_inputs.cpu().detach().numpy()[0, 0]
        pred = test_pred.cpu().detach().numpy()[0, 0]
        target = test_labels.cpu().detach().numpy()[0, 0]

        saved_input_name = save_input_path + list_labels[step].split('/')[-1][0:7] + '_ls.nii.gz'
        saved_pred_name = save_pred_path + list_labels[step].split('/')[-1][0:7] + '_pred.nii.gz'
        saved_gt_name = save_gt_path + list_labels[step].split('/')[-1][0:7] + '_hs.nii.gz'

        saved_preprocessed(input, sitk_imgs.GetOrigin(), sitk_imgs.GetDirection(), sitk_imgs.GetSpacing(), saved_input_name)
        saved_preprocessed(pred, sitk_imgs.GetOrigin(), sitk_imgs.GetDirection(), sitk_imgs.GetSpacing(), saved_pred_name)
        saved_preprocessed(target, sitk_imgs.GetOrigin(), sitk_imgs.GetDirection(), sitk_imgs.GetSpacing(), saved_gt_name)
        step += 1
        kkk += 1
    print('SSIM:{:.5f},  PSNR:{:.5f}.'.format(running_ssim / (kkk + 1), running_psnr / (kkk + 1)))


if __name__ == "__main__":
    test()


    
    


