import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class UKLPDDataset(Dataset):
    """Dataset for license plate super resolution training."""
    
    def __init__(self, image_paths, patch_size=64, scale_factor=8):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def apply_degradations(self, hr_patch):
        hr_patch_np = hr_patch.permute(1, 2, 0).numpy() * 255.0
        hr_patch_np = hr_patch_np.astype(np.uint8)
        
        lr_size = (self.patch_size // self.scale_factor, 
                   self.patch_size // self.scale_factor)
        lr_patch = cv2.resize(hr_patch_np, lr_size, interpolation=cv2.INTER_CUBIC)
        
        lr_patch = self.transform(lr_patch)
        hr_patch = self.transform(hr_patch_np)
        return lr_patch, hr_patch

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img_tensor = self.transform(img)
        
        h, w = img_tensor.shape[1:3]
        if h < self.patch_size or w < self.patch_size:
            raise ValueError("Image too small for patch size")
        
        i = np.random.randint(0, h - self.patch_size + 1)
        j = np.random.randint(0, w - self.patch_size + 1)
        hr_patch = img_tensor[:, i:i+self.patch_size, j:j+self.patch_size]
        
        lr_patch, hr_patch = self.apply_degradations(hr_patch)
        return lr_patch, hr_patch