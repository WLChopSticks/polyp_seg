import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import cv2

class poly_seg(Dataset):

    def __init__(self, root, csv_file, img_transform=None, mask_transform=None, iter_time=None):
        self.root = root
        img_mask = pd.read_csv(csv_file)
        self.imgs = img_mask['image'].values.tolist()
        self.masks = img_mask['mask'].values.tolist()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.iter_time = iter_time

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, 'images', self.imgs[idx]))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.iter_time is not None and self.iter_time != 0:
            mask = Image.open(os.path.join(self.root, 'masks', str(self.iter_time), self.masks[idx]))
        else:
            mask = Image.open(os.path.join(self.root, 'masks', self.masks[idx]))
        if mask.mode != 'L':
            mask = mask.convert('L')
        mask_arr = np.array(mask)
        mask_arr[mask_arr > 128] = 255
        mask_arr[mask_arr <= 128] = 0
        mask = Image.fromarray(mask_arr)

        seed = np.random.randint(1000000)
        if self.img_transform:
            img = self.img_transform(img, seed)
        if self.mask_transform:
            mask = self.mask_transform(mask, seed, mask_flag=True)

        img = img.sub(0.5).div(0.5)
        mask = mask.long().squeeze(dim=0)
        # mask = mask.sub(0.5).div(0.5)
        return img, mask

    def __len__(self):
        return len(self.imgs)
