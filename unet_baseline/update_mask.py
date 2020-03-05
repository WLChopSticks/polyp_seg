from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import Resize, ToTensor

def updateMask(train_image_dir, train_csv_path, mask_dir, model, index):
    '''
    1. 输入图像和分割后的结果
    2. 依据超像素方法将分割后的结果向后缩
    3. 将缩好的图片存储起来作为新的mask
    4. 读入新的mask,
    5. 重新开始训练
    '''
    test_img_aug = Compose_own([Resize(size=(288, 384)), ToTensor()])
    test_mask_aug = Compose_own([Resize(size=(288, 384)), ToTensor()])

    # model = unet_plus.NestedUNet(input_channels=3, output_channels=2, deepsupervision=False)
    model.load_state_dict(checkpoint['net'])
    train_dataset = poly_seg(root=train_image_dir, csv_file=train_csv_path, img_transform=test_img_aug,
                           mask_transform=test_mask_aug)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for input, label in train_dataloader:
        input = input.to(device)
        label = label.to(device)
        new_mask = np.zeros_like(label)
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            output = model(input)
            output = 255 * output.argmax(dim=1)
            image = input.cpu().numpy().astype(np.double).transpose(1, 2, 0)
            spixels = slic(image, n_segments=300, sigma=5)

            out = mark_boundaries(image * 255, spixels)
            cv2.imwrite('1.jpg', input * 255)
            cv2.imwrite('2.jpg', out)

            for (i, segVal) in enumerate(np.unique(spixels)):
                obj_tem = np.zeros(input.shape[:2], dtype="uint8")
                obj_tem[spixels == segVal] = 255
                overlap = obj_tem + output
                overlap[overlap > 255] = 255
                total_size = overlap.sum()
                if total_size - output.sum() <= 0:
                    new_mask = new_mask + obj_tem
            cv2.imwrite('3.jpg', new_mask)
            mask_name = mask_dir.split('/')[-1]
            mask_folder = mask_dir.split('/')[:,-2]
            import os
            new_mask_dir = os.path.join(mask_folder, str, mask_name)
            cv2.imwrite(new_mask_dir, new_mask)


