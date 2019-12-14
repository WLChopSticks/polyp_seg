import argparse
import os
import numpy as np
import torch
from datasets_own import poly_seg, Compose_own
from models import UNet
from PIL import Image

from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor
from utils import dice_fn


def valdata(val_csv_path, dataset_root, checkpoint_path):
    test_img_aug = Compose_own([Resize(size=(img_size, img_size)), ToTensor()])
    test_mask_aug = Compose_own([Resize(size=(img_size, img_size)), ToTensor()])
    # 加载参数
    checkpoint = torch.load(checkpoint_path)
    model = UNet(colordim=3, n_classes=2)
    model.load_state_dict(checkpoint['net'])
    val_dataset = poly_seg(root=dataset_root, csv_file=val_csv_path, img_transform=test_img_aug, mask_transform=test_mask_aug)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    val_dice = 0
    for inputs, labels in val_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            dice = dice_fn(outputs, labels)
            val_dice += dice.item()
    val_dice_epoch = val_dice / len(val_dataloader.dataset)
    print(val_dice_epoch)
    return val_dice_epoch

import sys
img_size = 256
dataset_root = os.path.join(sys.path[0],'../data/CVC-912/test')
val_csv_path = [os.path.join(sys.path[0],'../data/fixed-csv/test.csv')]
checkpoint_path = [os.path.join(sys.path[0],'../unet_baseline/checkpoint/0unet_params.pkl')]
dice = []
for i, j in zip(val_csv_path, checkpoint_path):
    dice.append(valdata(i, dataset_root, j))
print(sum(dice)/len(val_csv_path))
