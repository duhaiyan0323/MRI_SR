import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomAffine
from PIL import Image
import numpy as np


class SynchronizedTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img1, img2):
        seed = np.random.randint(2147483647)  # use the same seed
        torch.manual_seed(seed)
        img1 = self.transform(img1)
        torch.manual_seed(seed)
        img2 = self.transform(img2)
        return img1, img2
    


class MRIDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, plane='coronal'):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.plane = plane
        self.lr_img = nib.load(self.lr_dir).get_fdata()
        self.hr_img = nib.load(self.hr_dir).get_fdata()
        self.lr_slices, self.hr_slices= self._get_slices()

        self.transforms = SynchronizedTransform(Compose([
            ToTensor(),
            Normalize(mean=[0.0], std=[1.0]),  # Normalize to [0, 1]
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(degrees=30),
            RandomAffine(degrees=0, scale=(0.8, 1.2))
        ]))
    
    def _get_slices(self):
        if self.plane == 'coronal':
            lr_slices = [self.lr_img[:, i, :] for i in range(self.lr_img.shape[1])]
            hr_slices = [self.hr_img[:, i, :] for i in range(self.hr_img.shape[1])]
            
        elif self.plane == 'sagittal':
            lr_slices = [self.lr_img[i, :, :] for i in range(self.lr_img.shape[0])]
            hr_slices = [self.hr_img[i, :, :] for i in range(self.hr_img.shape[0])]
            
        else:
            raise ValueError("Plane must be 'coronal' or 'sagittal'")
        
        return lr_slices, hr_slices
    

    def __len__(self):
        return len(self.lr_slices)
        

    def __getitem__(self, idx):

        lr_slice = self.lr_slices[idx]
        #lr_slices = np.expand_dims(lr_slices, axis=0)  # Add channel dimension
        hr_slice = self.hr_slices[idx]
        #hr_slices = np.expand_dims(hr_slices, axis=0)  # Add channel dimension
        
        
        lr_slice = lr_slice.astype(np.float32)
        hr_slice = hr_slice.astype(np.float32)
        lr_slice, hr_slice = self.transforms(lr_slice, hr_slice)
        
        return lr_slice, hr_slice



# # 设置数据集目录
# lr_dir = '/home/hydu/MRI_SR/processMRI2/FLAIR/data/LR_Nearest/'
# hr_dir = '/home/hydu/MRI_SR/processMRI2/FLAIR/data/SR_Nearest/'

# batch_size = 4

# lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
# hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
# print(len(lr_files))
# for epoch in range(1):
#     for f_i in range(len(lr_files)):
#         lr_path = os.path.join(lr_dir,lr_files[f_i])
#         hr_path = os.path.join(hr_dir,hr_files[f_i])
#         print(lr_path)
#         coronal_dataset = MRIDataset(lr_path, hr_path, plane='coronal')
#         sagittal_dataset = MRIDataset(lr_path, hr_path, plane='sagittal')
#         coronal_loader = DataLoader(coronal_dataset, batch_size=batch_size, shuffle=True)
#         sagittal_loader = DataLoader(sagittal_dataset, batch_size=batch_size, shuffle=True)
#         print(len(coronal_loader))
#         for lr_coronal, hr_coronal in coronal_loader:

#             print("Low Resolution Coronal Shape:", lr_coronal.shape)
#             print("High Resolution Coronal Shape:", hr_coronal.shape)

#             # 假设您有一个名为tensor_data 的张量数据，数据范围在0到255之间
#             # 转换张量数据为numpy数组
#             tensor_data_np = lr_coronal[0,0,:,:].numpy()
#             tensor_data_np2 = hr_coronal[0,0,:,:].numpy()

#             # 将数据转换为8位整数
#             tensor_data_uint8 = (tensor_data_np).astype(np.uint8)
#             tensor_data2_uint8 = (tensor_data_np2).astype(np.uint8)

#             # 创建一个PIL图像对象
#             image = Image.fromarray(tensor_data_uint8)
#             image2 = Image.fromarray(tensor_data2_uint8)
#             np.save('lr_coronal.npy',tensor_data_np)
#             # 保存图像
#             image.save("lr_coronal.jpg")
#             image2.save("hr_coronal.jpg")