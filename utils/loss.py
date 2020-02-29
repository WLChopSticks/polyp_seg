import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from sklearn.metrics import jaccard_similarity_score
import numpy as np
from typing import cast

class UnionLossWithCrossEntropyAndDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(UnionLossWithCrossEntropyAndDiceLoss, self).__init__()
        self.crossEntropyLoss = CrossEntropyLoss2d(weight, size_average, ignore_index)
        self.diceLoss = DiceLoss(weight, size_average)

    def forward(self, inputs, targets):
        loss_crossEntropy = self.crossEntropyLoss(inputs, targets)
        loss_dice = self.diceLoss(inputs, targets)

        return loss_crossEntropy + loss_dice


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, threshold=0.5):
        N = targets.size(0)
        smooth = 1
        inputs = F.softmax(inputs, dim=1)
        inputs_obj = inputs[:, 1, :, :]
        inputs_obj[inputs_obj >= threshold] = 1
        inputs_obj[inputs_obj < threshold] = 0
        inputs_obj = inputs_obj.long()

        input_flat = inputs_obj.view(N, -1)
        target_flat = targets.view(N, -1)

        dice_coef = (2. * (input_flat * target_flat).float().sum() + smooth) / (input_flat.float().sum() + target_flat.float().sum() + smooth)
        loss = 1 - dice_coef
        return loss

class UnionLossWithCrossEntropyAndSize(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(UnionLossWithCrossEntropyAndSize, self).__init__()
        self.crossEntropyLoss = CrossEntropyLoss2d(weight, size_average, ignore_index)
        self.sizeLoss = Size_Loss_naive()

    def forward(self, inputs, targets):
        loss_crossEntropy = self.crossEntropyLoss(inputs, targets)
        loss_size = self.sizeLoss(inputs, targets)

        return loss_crossEntropy + loss_size
# ######## ------ Size loss function  (naive way) ---------- ###########
# --- This function will push the prediction to be close ot sizeGT ---#
class Size_Loss_naive():
    """
        Behaviour not exactly the same ; original numpy code used thresholding.
        Not quite sure it will have an inmpact or not
    """
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        super(Size_Loss_naive, self).__init__()
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, inputs, targets):
        print(inputs)
        inputs = F.log_softmax(inputs, dim=1)
        print(inputs)
        assert self.simplex(inputs)

        _, _, h, w = inputs.shape
        pred_size = einsum("bcwh->bc", input)[:, 1]
        target_size = 7845

        loss = (pred_size - target_size) ** 2
        loss = loss / (w * h)

        return loss / 100

    def simplex(t: Tensor, axis=1) -> bool:
        _sum = cast(Tensor, t.sum(axis).type(torch.float32))
        _ones = torch.ones_like(_sum, dtype=torch.float32)
        return torch.allclose(_sum, _ones)



def dice_fn(inputs, targets, threshold=0.5):
    inputs = F.softmax(inputs, dim=1)
    inputs_obj = inputs[:, 1, :, :]
    inputs_obj[inputs_obj>=threshold] = 1
    inputs_obj[inputs_obj<threshold] = 0
    dice = 0.
    smooth = 1
    for input_, target_ in zip(inputs_obj, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        intersection = (iflat * tflat).sum()
        dice_single = ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        dice += dice_single
    return dice

def IoU_fn(inputs, targets, threshold=0.5):
    inputs = F.softmax(inputs, dim=1)
    inputs_obj = inputs[:, 1, :, :]
    inputs_obj[inputs_obj >= threshold] = 1
    inputs_obj[inputs_obj < threshold] = 0
    IoU = 0.
    for input_, target_ in zip(inputs_obj, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        intersection = (iflat * tflat).sum()
        IoU_single = intersection / (iflat.sum() + tflat.sum() - intersection)
        IoU += IoU_single
    return IoU