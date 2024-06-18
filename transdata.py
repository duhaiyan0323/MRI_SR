import os
import shutil
import SimpleITK as sitk
import warnings
import glob
import numpy as np

def saved_preprocessed(savedImg, origin, direction, xyz_thickness, saved_name):
    newImg = sitk.GetImageFromArray(savedImg)
    newImg.SetOrigin(origin)
    newImg.SetDirection(direction)
    newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
    sitk.WriteImage(newImg, saved_name)

root = '/home/hydu/MRI_SR/processMRI/T1/SR/'
save_root = '/home/hydu/MRI_SR/processMRI/T1/SR2/'
dir_list=sorted(os.listdir(root))

for i in range(len(dir_list)):
    nii_file = os.path.join(root, dir_list[i])
    print(nii_file)
    sitk_img = sitk.ReadImage(nii_file)
    img = sitk.GetArrayFromImage(sitk_img)
    
    d,h,w = img.shape
    if d >=160:
        new_img = img[0:160]
    else:
        new_img = np.zeros((160,h,w))
        new_img[0:d] = img
    print(new_img.shape)
    saved_name  = os.path.join(save_root, dir_list[i])
    saved_preprocessed(new_img, sitk_img.GetOrigin(), sitk_img.GetDirection(), sitk_img.GetSpacing(), saved_name)