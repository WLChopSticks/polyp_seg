from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from datasets_own import poly_seg, Compose_own
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import Resize, ToTensor
import pandas as pd
from PIL import Image
from torchvision.transforms import Resize, ToTensor, ToPILImage, Compose
import os

def updateMask(train_image_dir, train_csv_path, mask_dir, model, index):
    '''
    1. 输入图像和分割后的结果
    2. 依据超像素方法将分割后的结果向后缩
    3. 将缩好的图片存储起来作为新的mask
    4. 读入新的mask,
    5. 重新开始训练
    '''
    test_img_aug = Compose([Resize(size=(288, 384)), ToTensor()])
    # test_mask_aug = Compose([Resize(size=(288, 384)), ToTensor()])

    dataframe = pd.read_csv(train_csv_path)
    new_mask_dir = os.path.join(train_image_dir, 'masks', str(index + 1))

    for j in range(len(dataframe)):

        series = dataframe.loc[j]
        root = train_image_dir
        image_name = series['image']
        gt_name = series['mask']
        image_path = os.path.join(root, 'images', image_name)
        img = Image.open(image_path)
        if index != 0:
            mask_path = os.path.join(root, 'masks', str(index), gt_name)
            new_mask_path = os.path.join(new_mask_dir, gt_name)
            gt = Image.open(mask_path)
        else:
            mask_path = os.path.join(root, 'masks', gt_name)
            new_mask_path = os.path.join(new_mask_dir, gt_name)
            gt = Image.open(mask_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if gt.mode != 'L':
            gt = gt.convert('L')
        gt_arr = np.array(gt)
        gt_arr[gt_arr > 128] = 255
        gt_arr[gt_arr <= 128] = 0
        gt = Image.fromarray(gt_arr)

        #网络预测mask
        mask_origin_size =predict_mask_with_net(img, test_img_aug,model)

        image = np.array(img)
        output = np.array(mask_origin_size)

        new_mask = np.zeros((image.shape[0],image.shape[1]), dtype="long")
        spixels = slic(image, n_segments=300, sigma=5)

        out = mark_boundaries(image, spixels)
        cv2.imwrite('1.jpg', image)
        cv2.imwrite('2.jpg', out * 255)
        cv2.imwrite('out.jpg',output)

        for (i, segVal) in enumerate(np.unique(spixels)):
            obj_tem = np.zeros(new_mask.shape[:2], dtype="long")
            obj_tem[spixels == segVal] = 255
            overlap = obj_tem + output
            overlap[overlap > 255] = 255
            total_size = overlap.sum()
            if total_size - output.sum() <= obj_tem.sum() * 0.5:
                new_mask = new_mask + obj_tem
        cv2.imwrite('new_out.jpg', new_mask)
        #如果新分割出的结果太小或者没有， 则用上一次的mask
        new_mask_size = new_mask.sum()
        origin_mask_size = np.array(gt).sum()
        ratio = new_mask_size / origin_mask_size
        #新的mask为0， 或者mask重叠区域过小或过大 超过30%， 则使用上一轮的mask送入下一轮训练
        if new_mask_size == 0 or (ratio < 0.7 or ratio > 1.3):
            new_mask = np.array(gt)
            print('use last mask for next train: ' + new_mask_path)

        cv2.imwrite('new_out2.jpg', new_mask)
        if not os.path.exists(new_mask_dir):
            os.mkdir(new_mask_dir)

        cv2.imwrite(new_mask_path, new_mask)
        print(str(j) + 'save new mask at: ', new_mask_path)

def predict_mask_with_net(image, transform, model):
    #输出网络的预测图，原尺寸大小
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_for_net = transform(image)
    img_for_net = img_for_net.to(device)
    img_for_net = img_for_net.sub(0.5).div(0.5)

    img_for_net = img_for_net.unsqueeze(dim=0)
    predict = model(img_for_net)
    to_pil = ToPILImage()
    mask = to_pil(torch.tensor(255 * predict.argmax(dim=1).detach(), dtype=torch.uint8))
    mask_origin_size = mask.resize(image.size)

    return mask_origin_size
