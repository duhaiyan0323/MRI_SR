import torch
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim
import numpy as np
import cv2
import ants
from skimage.metrics.simple_metrics import peak_signal_noise_ratio
import random
import SimpleITK as sitk
import os

def saved_preprocessed(savedImg, origin, direction, xyz_thickness, saved_name):
    newImg = sitk.GetImageFromArray(savedImg)
    newImg.SetOrigin(origin)
    newImg.SetDirection(direction)
    newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
    sitk.WriteImage(newImg, saved_name)

def psnr_np(pred, true):
    pred = pred.numpy()
    true = true.numpy()
    return 10. * np.log10(1. / np.mean((pred-true)**2))
    #return peak_signal_noise_ratio(pred, true)

def psnr(pred, true):
    return 10. * torch.log10(1. / torch.mean((pred-true)**2))


def ssim_3d_all(pred, target):
    pred = pred.numpy()
    target = target.numpy()
    
    pred = pred.transpose(2,0,1)
    target = target.transpose(2,0,1)
    
    pred_batch = pred[np.newaxis,:,:,:]
    target_batch = target[np.newaxis,:,:,:]

    pred_batch = torch.from_numpy(pred_batch).float()
    target_batch = torch.from_numpy(target_batch).float()
    ssim_value = ssim(pred_batch, target_batch, data_range=1, size_average=True)
    ms_ssim_value = ms_ssim(pred_batch, target_batch, data_range=1, size_average=True)
        
    return ssim_value, ms_ssim_value

def ssim_3d(pred, target):
    ssim_total = 0.
    ms_ssim_total = 0.
    b = pred.shape[0]
    for i in range(b):
        pred_batch = pred[i]
        target_batch = target[i]
        ssim_val = ssim(pred_batch, target_batch, data_range=1, size_average=True)
        #ms_ssim_val = ms_ssim(pred_batch, target_batch, data_range=1, size_average=True)
        ssim_total += ssim_val
        #ms_ssim_total += ms_ssim_val
    
    return ssim_total/b#, ms_ssim_total/b
