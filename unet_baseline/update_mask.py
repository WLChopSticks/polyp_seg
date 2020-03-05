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

def updateMask(train_image_dir, train_csv_path, mask_dir, model, index):
    '''
    1. 输入图像和分割后的结果
    2. 依据超像素方法将分割后的结果向后缩
    3. 将缩好的图片存储起来作为新的mask
    4. 读入新的mask,
    5. 重新开始训练
    '''
    test_img_aug = Compose([Resize(size=(288, 384)), ToTensor()])
    test_mask_aug = Compose([Resize(size=(288, 384)), ToTensor()])

    dataframe = pd.read_csv(train_csv_path)
    for j in range(len(dataframe)):

        series = dataframe.loc[j]
        root = train_image_dir
        image_name = series['image']
        gt_name = series['mask']
        image_path = os.path.join(root, 'images', image_name)
        img = Image.open(image_path)
        if index != 0:
            mask_path = os.path.join(root, 'masks', str(index), gt_name)
            new_mask_dir = os.path.join(root, 'masks', str(index + 1))
            new_mask_path = os.path.join(root, 'masks', str(index + 1), gt_name)
            gt = Image.open(mask_path)
        else:
            mask_path = os.path.join(root, 'masks', gt_name)
            new_mask_dir = os.path.join(root, 'masks', str(1))
            new_mask_path = os.path.join(root, 'masks', str(1), gt_name)
            gt = Image.open(mask_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if gt.mode != 'L':
            gt = gt.convert('L')
        gt_arr = np.array(gt)
        gt_arr[gt_arr > 128] = 255
        gt_arr[gt_arr <= 128] = 0
        gt = Image.fromarray(gt_arr)

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_for_net = test_img_aug(img)
        img_for_net = img_for_net.to(device)
        img_for_net = img_for_net.sub(0.5).div(0.5)

        img_for_net = img_for_net.unsqueeze(dim=0)
        predict = model(img_for_net)
        to_pil = ToPILImage()
        mask = to_pil(torch.tensor(255 * predict.argmax(dim=1).detach(), dtype=torch.uint8))
        mask_origin_size = mask.resize(gt.size)


        mask = 255 * predict.argmax(dim=1)
        resize_origin = Compose([Resize(size=gt.size)])
        #mask = resize_origin(mask)
        mask = np.array(mask_origin_size)
        #mask = mask.squeeze(0).cpu().numpy()

        image = cv2.imread(image_path)
        #image = cv2.resize(image,(288,384))

        image = np.array(img)
        output = np.array(mask_origin_size)
        #如果mask没有分出来， 则用上一次迭代的mask
        if output.sum() == 0:
            output = getLastMask(root, gt_name, index)

        #image = input[0].cpu().numpy().astype(np.double).transpose(1, 2, 0)
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
            if total_size - output.sum() <= 0:
                new_mask = new_mask + obj_tem
        cv2.imwrite('new_out.jpg', new_mask)
        #如果新分割出的结果太小或者没有， 则用上一次的mask
        new_mask_size = new_mask.sum()
        origin_mask_size = np.array(gt).sum()
        ratio = (origin_mask_size-new_mask_size) / origin_mask_size
        if new_mask_size == 0 or ratio < 50:
            new_mask = np.array(gt)

        cv2.imwrite('new_out2.jpg', new_mask)
        if not os.path.exists(new_mask_dir):
            os.mkdir(new_mask_dir)

        cv2.imwrite(new_mask_path, new_mask)
        print(str(j) + 'save new mask at: ', new_mask_path)
import os
def getLastMask(root, gt_name,index):
    print('get last mask for ' + gt_name)
    if index <= 1:
        last_mask_path = os.path.join(root, 'masks', gt_name)
    else:
        last_mask_path = os.path.join(root, 'masks', str(index), gt_name)
    last_mask = Image.open(last_mask_path)
    if last_mask.mode != 'L':
        last_mask = last_mask.convert('L')
    output = np.array(last_mask)
    return output

