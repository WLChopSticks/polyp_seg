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
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    val_dice = 0
    Sensitivity = 0
    Specificity = 0
    Precision = 0
    F1 = 0
    F2 = 0
    ACC_overall = 0
    IoU_poly = 0
    IoU_bg = 0
    IoU_mean = 0

    for inputs, labels in val_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            dice = dice_fn(outputs, labels)
            val_dice += dice.item()

            predict_masks = torch.argmax(outputs)
            #
            validate_gt = labels

            label_probs_rep_inverse = (predict_masks == 0).float()
            train_label_rep_inverse = (validate_gt == 0).float()

            # calculate TP, FP, TN, FN
            TP = predict_masks.mul(validate_gt).sum()
            FP = predict_masks.mul(train_label_rep_inverse).sum()
            TN = label_probs_rep_inverse.mul(train_label_rep_inverse).sum()
            FN = label_probs_rep_inverse.mul(validate_gt).sum()

            if TP.item() == 0:
                # print('TP=0 now!')
                # print('Epoch: {}'.format(epoch))
                # print('i_batch: {}'.format(i_batch))

                TP = torch.Tensor([1]).cuda()

            # Sensitivity, hit rate, recall, or true positive rate
            temp_Sensitivity = TP / (TP + FN)

            # Specificity or true negative rate
            temp_Specificity = TN / (TN + FP)

            # Precision or positive predictive value
            temp_Precision = TP / (TP + FP)

            # F1 score = Dice
            temp_F1 = 2 * temp_Precision * temp_Sensitivity / (temp_Precision + temp_Sensitivity)

            # F2 score
            temp_F2 = 5 * temp_Precision * temp_Sensitivity / (4 * temp_Precision + temp_Sensitivity)

            # Overall accuracy
            temp_ACC_overall = (TP + TN) / (TP + FP + FN + TN)

            # Mean accuracy
            # temp_ACC_mean = TP / pixels

            # IoU for poly
            temp_IoU_poly = TP / (TP + FP + FN)

            # IoU for background
            temp_IoU_bg = TN / (TN + FP + FN)

            # mean IoU
            temp_IoU_mean = (temp_IoU_poly + temp_IoU_bg) / 2.0

            # To Sum
            Sensitivity += temp_Sensitivity.item()
            Specificity += temp_Specificity.item()
            Precision += temp_Precision.item()
            F1 += temp_F1.item()
            F2 += temp_F2.item()
            ACC_overall += temp_ACC_overall.item()
            IoU_poly += temp_IoU_poly.item()
            IoU_bg += temp_IoU_bg.item()
            IoU_mean += temp_IoU_mean.item()


    val_dice_epoch = val_dice / len(val_dataloader.dataset)
    print('val_dice ' + val_dice_epoch)
    val_Sensitivity_epoch = Sensitivity / len(val_dataloader.dataset)
    print('val_Sensitivity ' + val_Sensitivity_epoch)



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
