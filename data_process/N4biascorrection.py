# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:13:49 2021

@author: wangdan
"""

#-*-coding:utf-8-*-
import os
import shutil
import SimpleITK as sitk
import warnings
import glob
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection
 
 
def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)
 
def normalize_image(in_file, out_file, bias_correction=True):
    if bias_correction:
        correct_bias(in_file, out_file)
    else:
        shutil.copy(in_file, out_file)
    return out_file

if __name__ == '__main__':
    
    root = '/home/hydu/MRI_SR/processMRI2/FLAIR/N4biascorrection/'
    save_root = '/home/hydu/MRI_SR/processMRI2/FLAIR/N4biascorrection/'
    dir_list=sorted(os.listdir(root))
    
    for i in range(len(dir_list)):
        nii_file = os.path.join(root, dir_list[i])
        n4_nii_file = os.path.join(save_root, dir_list[i].replace('.nii.gz', '.nii.gz'))
        correct_bias(nii_file, n4_nii_file)
        print(dir_list[i])
                               