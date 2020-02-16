import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import jaccard_similarity_score
import numpy as np

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

def dice_fn(inputs, targets, threshold=0.5):
    inputs = F.softmax(inputs, dim=1)
    inputs_obj = inputs[:, 1, :, :]
    inputs_obj[inputs_obj>=threshold] = 1
    inputs_obj[inputs_obj<threshold] = 0
    dice = 0.
    for input_, target_ in zip(inputs_obj, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        intersection = (iflat * tflat).sum()
        dice_single = ((2. * intersection) / (iflat.sum() + tflat.sum()))
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