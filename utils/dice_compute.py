import os
import cv2
'''
计算两个文件夹中图片的dice
'''
gt_dir = '/Users/wanglei/Documents/polyp_seg/polyp_seg/unet_baseline/checkpoint/deeplabV3+/output/new_gt/masks'
mask_dir = '/Users/wanglei/Documents/polyp_seg/polyp_seg/unet_baseline/checkpoint/deeplabV3+/output/new_gt/ite1_gt'

for (root, dirs, files) in os.walk(mask_dir):
    dice_total = 0
    for file in files:

        mask_path = os.path.join(root, file)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask / 255

        # if '612' in file:
        #     file = file.replace('bmp', 'tif')
        print(file)
        gt_path = os.path.join(gt_dir, file)
        gt = cv2.imread(gt_path)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        gt = gt / 255

        intersection = (mask * gt).sum()
        dice_single = ((2. * intersection) / (mask.sum() + gt.sum()))
        dice_total += dice_single

    dice = dice_total / len(files)
    print(dice)





