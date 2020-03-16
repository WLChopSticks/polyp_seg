import argparse
import os
import sys
import numpy as np
import torch
from datasets_own import poly_seg, Compose_own
from models import UNet, unet_plus
from models.deeplab3_plus.deeplab import *
from PIL import Image

from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor
from utils import dice_fn


def validate(val_csv_path, dataset_root, checkpoint_path):
    test_img_aug = Compose_own([Resize(size=(288, 384)), ToTensor()])
    test_mask_aug = Compose_own([Resize(size=(288, 384)), ToTensor()])
    # 加载参数
    checkpoint = torch.load(checkpoint_path)
    # model = UNet(colordim=3, n_classes=2)

    model = DeepLab(num_classes=2,
                     backbone='resnet',
                     output_stride=16,
                     sync_bn=True,
                     freeze_bn=False)

    #model = unet_plus.NestedUNet(input_channels=3, output_channels=2, deepsupervision=False)
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

    for inputs, labels, _ in val_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            dice = dice_fn(outputs, labels)
            val_dice += dice.item()

            predict_masks = torch.argmax(outputs, dim=1).float()
            #
            validate_gt = labels.float()

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

    Rec = Sensitivity / len(val_dataloader.dataset)
    Spec = Specificity / len(val_dataloader.dataset)
    Prec = Precision / len(val_dataloader.dataset)
    Dice = F1 / len(val_dataloader.dataset)
    F2 = F2 / len(val_dataloader.dataset)
    IoU_p = IoU_poly / len(val_dataloader.dataset)
    IoU_b = IoU_bg / len(val_dataloader.dataset)
    IoU_m = IoU_mean / len(val_dataloader.dataset)
    Acc = ACC_overall / len(val_dataloader.dataset)

    print('val_dice ' + str(val_dice / len(val_dataloader.dataset)))
    print('Recall: %f ' % Rec)
    print('Specificity: %f '% Spec)
    print('Precision: %f ' % Prec)
    print('Dice: %f ' % Dice)
    print('F2: %f ' % F2)
    print('IoU_p: %f ' % IoU_p)
    print('IoU_b: %f ' % IoU_b)
    print('IoU_m: %f ' % IoU_m)
    print('Acc: %f ' % Acc)

    result = {'val_dice' : str(val_dice / len(val_dataloader.dataset)),
              'Recall'   : str(Rec),
              'Specificity': str(Spec),
              'Precision' : str(Prec),
              'Dice' : str(Dice),
              'F2' : str(F2),
              'IoU_p' : str(IoU_p),
              'IoU_b' : str(IoU_b),
              'IoU_m' : str(IoU_m),
              'Acc' : str(Acc),}

    return result

if __name__ == "__main__":

    dataset_root = os.path.join(sys.path[0],'../data/CVC-912/test')
    val_csv_path = os.path.join(sys.path[0],'../data/fixed-csv/test.csv')
    checkpoint_path = os.path.join(sys.path[0],'../unet_baseline/checkpoint/deeplabV3+/0init_0.7668.pkl')
    result = validate(val_csv_path, dataset_root, checkpoint_path)
    print(result)
    # send result to wechat
    import requests

    url = "https://sc.ftqq.com/SCU28703Te109f3ff3fede315f4017d79786274ab5b35cf275612b.send?"
    url2 = "https://sc.ftqq.com/SCU87403Tdd9ec9b4572930aee59a144326d0f5e15e5c9a4163f6a.send?"

    result_str = 'val_dice: {0}\n\n' \
                 'Recall: {1}\n\n' \
                 'Specificity: {2}\n\n' \
                 'Precision: {3}\n\n' \
                 'Dice: {4}\n\n' \
                 'F2: {5}\n\n' \
                 'IoU_p: {6}\n\n' \
                 'IoU_b: {7}\n\n' \
                 'IoU_m: {8}\n\n' \
                 'Acc: {9}\n\n'.format(result['val_dice'],result['Recall'],result['Specificity']
                                     ,result['Precision'],result['Dice'],result['F2'],result['IoU_p'],
                                     result['IoU_b'],result['IoU_m'],result['Acc'],)

    params = {"text": 'linux: ' + 'test',
              'desp': result_str + '\n\nthe infomation is to wl'}

    res = requests.get(url=url, params=params)
    params2 = {"text": 'ubuntu: ' + 'test',
              'desp': result_str + '\n\nthis message is to ljx'}
    res2 = requests.get(url=url2, params=params2)
    print(res.text)
    print(res2.text)


