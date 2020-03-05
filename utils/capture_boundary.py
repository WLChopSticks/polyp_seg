from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import cv2
import numpy as np
import torch
from torch.autograd import Variable


def get_boundary(images, inputs, targets, threshold):
    n = targets.size(0)
    inputs_obj_pred = inputs[:, 1, :, :]
    inputs_obj = inputs[:, 1, :, :].clone()
    inputs_obj[inputs_obj >= threshold] = 1
    inputs_obj[inputs_obj < threshold] = 0
    inputs_obj = inputs_obj.long()
    first = True
    for j in range(n):
        one_image = images[j].cpu().numpy().astype(np.double).transpose(1,2,0)
        one_input = inputs_obj[j].cpu().numpy().astype(np.uint8)
        one_target = targets[j].cpu().numpy() * 255
        base_mask = np.zeros_like(one_target, dtype="uint8")
        spixels = slic(one_image, n_segments=300, sigma=5)
        out = mark_boundaries(one_image*255, spixels)
        cv2.imwrite('1.jpg',one_image*255)
        cv2.imwrite('2.jpg',out)
        cv2.imwrite('3.jpg', one_target*255)

        for (i, segVal) in enumerate(np.unique(spixels)):
            obj_tem = np.zeros(one_input.shape[:2], dtype="uint8")
            obj_tem[spixels == segVal] = 255
            overlap = obj_tem + one_input * 255
            overlap[overlap >255] = 255
            total_size = overlap.sum()
            if total_size - one_input.sum() * 255 <= 0:
                base_mask = base_mask + obj_tem
        sboundary = cv2.Canny(base_mask, 30, 70)
        iboundary = cv2.Canny(one_input * 255, 30, 70)
        cv2.imwrite('4.jpg', sboundary)
        cv2.imwrite('5.jpg', iboundary)
        sboundary[sboundary == 255] = 1
        one_input_pred = inputs_obj_pred[j].cpu()
        ipred = one_input_pred * torch.from_numpy(sboundary).float()
        #ipred = one_input_pred * torch.from_numpy(sboundary).float()
        #cv2.imwrite('6.jpg', ipred.detach().numpy()*255)
        if first:
            super_boundary = torch.from_numpy(sboundary)
            input_pred = ipred
            first = False
            input_pred = torch.unsqueeze(torch.unsqueeze(ipred, 0), 0)
            input_pred = torch.cat((input_pred, torch.ones_like(input_pred)), 1)
            super_boundary = torch.unsqueeze(super_boundary, 0)
        else:
            ipred = torch.unsqueeze(torch.unsqueeze(ipred, 0), 0)
            ipred = torch.cat((ipred, torch.zeros_like(ipred)), 1)
            sboundary = torch.unsqueeze(torch.from_numpy(sboundary), 0)
            super_boundary = torch.cat((super_boundary, sboundary),dim=0)
            input_pred = torch.cat((input_pred, ipred), dim=0)

    return input_pred.float(), super_boundary.long()