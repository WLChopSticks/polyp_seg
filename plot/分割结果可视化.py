import os
import numpy as np
import torch
import pandas as pd
from utils.dcm_nii_ops import load_dicom, load_nii
from PIL import Image
from datasets.joint_transform import Compose, Resize, ToTensor, Normalize, RandomRotate, RandomHorizontallyFlip
from models.unet import UNet
from matplotlib import pyplot as plt
import cv2


def get_img_gt_result_draw(dataset_root=None, dataframe=None, index=None, transform=None,
                           savedir=None, draw_contour=False, model=None):

    series = dataframe.loc[index]
    root = dataset_root  # 数据根目录, 比如：/home/dell/jx/data/直肠癌/T1CEAXI_115
    img_mask = series
    patient_id = img_mask['patient_id']
    dcm_name = img_mask['dcm_name']
    nii_index = img_mask['nii_index']

    dcm_path = os.path.join(root, patient_id, dcm_name)

    dcm_array = load_dicom(dcm_path)
    dcm_array = np.uint8((dcm_array / dcm_array.max()) * 255)
    img = Image.fromarray(dcm_array, mode='L')
    if img.mode != 'RGB':
        img = img.convert('RGB')

    nii_array = load_nii(os.path.join(root, patient_id), nii_index)
    nii_array = np.uint8(nii_array * 255)
    gt = Image.fromarray(nii_array, mode='L')

    # 得到预测值
    mask_origin_size = forward_vis(model, image=img, gt=gt, transform=transform, visualize=False)
    savepath = os.path.join(savedir, patient_id.split('/')[-1]+'_'+dcm_name+'_cp.png')
    if draw_contour:
        draw_edge_on_image(gt, mask_origin_size, img, savepath, show=False)
    return img, gt, mask_origin_size


def forward_vis(model=None, image=None, gt=None, transform=None, visualize=False):
    model.eval()
    gt_for_net = gt
    img_for_net, gt_for_net = transform(image, gt_for_net)
    img_for_net = img_for_net.unsqueeze(dim=0)
    predict = model(img_for_net)
    mask = np.uint8(np.squeeze(predict.argmax(dim=1).detach().numpy(), axis=0)) * 255
    mask = Image.fromarray(mask)
    mask_origin_size = mask.resize(gt.size)

    if visualize:
        plt.figure()
        plt.subplot(221)
        plt.imshow(img)
        plt.title('原图')
        plt.subplot(222)
        plt.imshow(gt, 'gray')
        plt.title('ground truth')
        plt.subplot(224)
        plt.imshow(mask_origin_size, 'gray')
        plt.title('predicted mask')
        plt.show()

    return mask_origin_size


def draw_edge_on_image(gt, result, image, savepath, show=False):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    gt = cv2.cvtColor(np.asarray(gt), cv2.COLOR_RGB2BGR)
    gt_edge = cv2.Canny(gt, 0, 0) / 255  # 单通道
    result = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
    result_edge = cv2.Canny(result, 0, 0) / 255  # 单通道
    #  红色的色号是(255,0,0)，绿色的色号是（0,255,0)，蓝色的色号是(0,0,255）
    # cv2图像模式是BGR
    gt_edge = np.transpose(np.array([gt_edge, gt_edge, gt_edge]), [1, 2, 0])
    result_edge = np.transpose(np.array([result_edge, result_edge, result_edge]), [1, 2, 0])
    gt_edge[..., 2] = gt_edge[..., 2] * 255
    result_edge[..., 1] = result_edge[..., 1] * 255
    gt_edge = gt_edge.astype(np.uint8)
    result_edge = result_edge.astype(np.uint8)
    overlapping = cv2.addWeighted(image, 1, gt_edge, 0.3, 0)
    overlapping = cv2.addWeighted(overlapping, 1, result_edge, 0.3, 0)
    cv2.imwrite(savepath, overlapping)
    if show:
        cv2.imshow('1', overlapping)
        cv2.waitKey()
        cv2.destroyAllWindows()


# 参数
# /home/dell/jx/Projects/utils/dcm_split_5fold/T2WI/val_data_split1.csv
# /home/dell/jx/Projects/utils/dcm_split_5fold/T2WIFS/val_data_split1.csv
# /home/dell/jx/data/直肠癌/T2WIFS_AXI_194
# /home/dell/jx/data/直肠癌/T2WI_Sag_194
# /home/dell/jx/Seg/分割结果可视化/t2wi /home/dell/jx/Seg/分割结果可视化/t2wifs
val_csv_path = '/home/dell/jx/Projects/utils/dcm_split_5fold/T2WI/val_data_split1.csv'
dataset_root = '/home/dell/jx/data/直肠癌/T2WI_Sag_194'
savedir = '/home/dell/jx/Seg/分割结果可视化/t2wi'
test_index = 100
img_size_to_net = 256
transform = Compose([Resize(size=(img_size_to_net, img_size_to_net)), ToTensor(), Normalize(mean=(0.5, 0.5, 0.5),
                                                                                            std=(0.5, 0.5, 0.5))])
# 加载参数
checkpoint = torch.load('/home/dell/jx/Seg/weights/调参后的baseline/T2WI_fold1_baseline256.pkl')
model = UNet(colordim=3, n_classes=2)
model.load_state_dict(checkpoint['net'])
dataframe = pd.read_csv(val_csv_path)


# 得到图片并进行预测
for i in range(len(dataframe)):
    get_img_gt_result_draw(dataset_root=dataset_root, dataframe=dataframe, index=i, transform=transform,
                           savedir=savedir, draw_contour=True, model=model)





