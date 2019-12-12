import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import jaccard_similarity_score
import numpy as np

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

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