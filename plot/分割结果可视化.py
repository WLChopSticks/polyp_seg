import os
import numpy as np
import torch
import pandas as pd
from torchvision.transforms import Resize, ToTensor, ToPILImage, Compose
from PIL import Image
from models.unet import UNet
from matplotlib import pyplot as plt
import cv2


def get_img_gt_result_draw(dataset_root=None, dataframe=None, index=None, img_transform=None,
                           savedir=None, draw_contour=False, model=None):

    series = dataframe.loc[index]
    root = dataset_root
    image_name = series['image']
    gt_name = series['mask']

    img = Image.open(os.path.join(dataset_root, 'images', image_name))
    gt = Image.open(os.path.join(dataset_root, 'masks', gt_name))

    if img.mode != 'RGB':
        img = img.convert('RGB')
    if gt.mode != 'L':
        gt = gt.convert('L')
    gt_arr = np.array(gt)
    gt_arr[gt_arr > 128] = 255
    gt_arr[gt_arr <= 128] = 0
    gt = Image.fromarray(gt_arr)

    # 得到预测值
    mask_origin_size = forward_vis(model, image=img, gt=gt, img_transform=img_transform, visualize=False)

    savepath = os.path.join(savedir, image_name.split('.')[0] + '_cp.png')
    if draw_contour:
        draw_on_image(gt, mask_origin_size, img, savepath, show=False)
    return img, gt, mask_origin_size


def forward_vis(model=None, image=None, gt=None, img_transform=None, visualize=False):
    model.eval()
    img_for_net = img_transform(image)
    img_for_net = img_for_net.to(device)
    img_for_net = img_for_net.sub(0.5).div(0.5)

    img_for_net = img_for_net.unsqueeze(dim=0)
    predict = model(img_for_net)
    to_pil = ToPILImage()
    mask = to_pil(torch.tensor(255*predict.argmax(dim=1).detach(), dtype=torch.uint8))
    mask_origin_size = mask.resize(gt.size)

    if visualize:
        plt.figure()
        plt.subplot(131)
        plt.imshow(image)
        plt.title('orig')
        plt.subplot(132)
        plt.imshow(gt, 'gray')
        plt.title('ground truth')
        plt.subplot(133)
        plt.imshow(mask_origin_size, 'gray')
        plt.title('predicted mask')
        plt.show()

    return mask_origin_size


def draw_on_image(gt, result, image, savepath, show=False):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    gt = cv2.cvtColor(np.asarray(gt), cv2.COLOR_RGB2BGR)
    result = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
    #  红色的色号是(255,0,0)，绿色的色号是（0,255,0)，蓝色的色号是(0,0,255）
    # cv2图像模式是BGR
    # gt = np.transpose(gt, [1, 2, 0])
    # result = np.transpose(result, [1, 2, 0])
    gt[..., 2] = gt[..., 2] * 255
    result[..., 1] = result[..., 1] * 255
    overlapping = cv2.addWeighted(image, 1, gt, 0.3, 0)
    overlapping = cv2.addWeighted(overlapping, 1, result, 0.3, 0)
    htich = np.hstack((image, overlapping))
    cv2.imwrite(savepath, htich)
    if show:
        cv2.imshow('1', overlapping)
        cv2.waitKey()
        cv2.destroyAllWindows()


# 参数
dataset_root = r'E:\datasets\CVC-912\test'
val_csv_path = r'E:\code\polyp_seg\data\fixed-csv\test.csv'
savedir = r'E:\code\polyp_seg\plot\results'
checkpoint = r'E:\code\polyp_seg\unet_baseline\checkpoint\0unet_params.pkl'
test_index = 100
img_size_to_net = 256
test_img_aug = Compose([Resize(size=(img_size_to_net, img_size_to_net)), ToTensor()])
test_mask_aug = Compose([Resize(size=(img_size_to_net, img_size_to_net)), ToTensor()])

# 加载参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint)
model = UNet(colordim=3, n_classes=2)
model.load_state_dict(checkpoint['net'])
model.to(device)
dataframe = pd.read_csv(val_csv_path)

# 得到图片并进行预测
for i in range(len(dataframe)):
    get_img_gt_result_draw(dataset_root=dataset_root, dataframe=dataframe, index=i, img_transform=test_img_aug,
                           savedir=savedir, draw_contour=True, model=model)
