import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomAffine

class MRIDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, plane='coronal'):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.plane = plane

        assert len(self.lr_files) == len(self.hr_files), "Mismatch between LR and HR datasets"

        self.transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1] (if necessary, adjust mean and std)
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(degrees=30),
            RandomAffine(degrees=0, scale=(0.8, 1.2))
        ])

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])

        lr_img = nib.load(lr_path).get_fdata()
        hr_img = nib.load(hr_path).get_fdata()

        if self.plane == 'coronal':
            lr_slices = [lr_img[:, i, :] for i in range(lr_img.shape[1])]
            hr_slices = [hr_img[:, i, :] for i in range(hr_img.shape[1])]
        elif self.plane == 'sagittal':
            lr_slices = [lr_img[i, :, :] for i in range(lr_img.shape[0])]
            hr_slices = [hr_img[i, :, :] for i in range(hr_img.shape[0])]
        else:
            raise ValueError("Plane must be 'coronal' or 'sagittal'")

        lr_slices = [self.transforms(slice.astype(np.float32)) for slice in lr_slices]
        hr_slices = [self.transforms(slice.astype(np.float32)) for slice in hr_slices]

        return torch.stack(lr_slices), torch.stack(hr_slices)
