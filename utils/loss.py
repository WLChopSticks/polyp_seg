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
        loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        return loss

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
        loss_size = torch.log10(loss_size)
        return loss_crossEntropy + 0.03 * loss_size

# ######## ------ Size loss function  (naive way) ---------- ###########
def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)
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
        inputs = F.softmax(inputs, dim=1)
        assert simplex(inputs)

        _, _, h, w = inputs.shape
        preds = inputs.sum(dim = [2,3])
        pred_size = preds[:,1]#get the object channel


        target_size = targets.sum(dim=[1,2],dtype=torch.float32)
        #ellipse size
        target_size *= 0.785

        loss = (pred_size - target_size) ** 2
        loss = loss.sum() / (w * h)

        #return 0
        return loss

class UnionLossWithCrossEntropyAndDiceAndBoundary(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(UnionLossWithCrossEntropyAndDiceAndBoundary, self).__init__()
        self.crossEntropyLoss = CrossEntropyLoss2d(weight, size_average, ignore_index)
        self.diceLoss = DiceLoss()
        self.boundLoss = Boundary_Loss()

    def forward(self, inputs, targets, starget, dice_co, boundary_co):
        loss_crossEntropy = self.crossEntropyLoss(inputs, targets)
        diceLoss = self.diceLoss(inputs, targets)
        loss_boundary = self.boundLoss(inputs, targets)

        return loss_crossEntropy +1 * diceLoss + loss_boundary

class Boundary_Loss(nn.Module):
    """
        Behaviour not exactly the same ; original numpy code used thresholding.
        Not quite sure it will have an inmpact or not
    """
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        super(Boundary_Loss, self).__init__()

    def __call__(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        assert simplex(inputs)
        preds = torch.argmax(inputs, dim=1)
        # preds = inputs[:, 1, :, :]
        loss = 0
        for i in range(len(inputs)):
            pred = preds[i]
            data2 = class2one_hot(pred, 2)
            # print(data2)
            data2 = data2[0].cpu().numpy()
            data3 = one_hot2dist(data2)  # bcwh

            logits = class2one_hot(targets[i], 2)

            Loss = SurfaceLoss()
            data3 = torch.tensor(data3).unsqueeze(0).to(device='cuda')

            res = Loss(logits, data3, None)
            loss += res

        return loss * loss

        gts = targets

        loss = torch.mean(torch.sum(gts * torch.log(gts / (preds+1e-8))))

        return loss

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


import torch
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res

def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            # print('negmask:', negmask)
            # print('distance(negmask):', distance(negmask))
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
            # print('res[c]', res[c])
    return res


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

    # Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu().detach()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]   #这里忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3

    # probs: bcwh, dist_maps: bcwh
    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        #print('pc', pc)
        #print('dc', dc)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss