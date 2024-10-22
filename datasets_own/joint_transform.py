import numbers
import random

from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
from torchvision.transforms import ColorJitter
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class Compose_own(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, seed, mask_flag=False):
        for t in self.transforms:
            if isinstance(t, ColorJitter) and mask_flag:
                continue
            random.seed(seed)
            img = t(img)
        return img

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert len(self.size) == 2
        ow, oh = self.size
        return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class ToPILImage(object):
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, img, mask):
        return Image.fromarray(img), Image.fromarray(mask)


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)



class ToTensor(object):
    """ puts channels in front and convert to float, except if mode palette
    """
    def __init__(self):
        pass

    def __call__(self, img, mask, num_classes=2):
        img = torch.from_numpy(np.array(img).transpose(2,0,1)).float() / 255.0
        mask = np.array(mask) # [:,:,np.newaxis]
        mask = torch.from_numpy(mask)
        mask = ((num_classes -1) * mask.float() / 255.0).long()
        return img, mask

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).unsqueeze(1).unsqueeze(2)
        self.std = torch.FloatTensor(std).unsqueeze(1).unsqueeze(2)

    def __call__(self, img, mask):

        if self.std is None:
            img = img.sub(self.mean)
        else:
            img = img.sub(self.mean).div(self.std)

        return img, mask

class Simple_Normalize(object):
    def __init__(self):
        pass

    def __call__(self, img, mask):
        return img.div(255), mask
