from torchvision.transforms import RandomAffine, RandomRotation, RandomHorizontalFlip, ColorJitter, Resize, ToTensor, Normalize
from PIL import Image
import random
import numpy as np
import torch

class Compose_own(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, seed, mask_flag=False):
        for t in self.transforms:
            if isinstance(ColorJitter, t) and mask_flag:
                continue
            random.seed(seed)
            img = t(img)
        return img

train_img_aug = Compose_own([
        RandomAffine(90, shear=45),
        RandomRotation(90),
        RandomHorizontalFlip(),
        ColorJitter(),
        Resize(size=(img_size, img_size)),
        ToTensor()])

train_mask_aug = Compose_own([Resize(size=(img_size, img_size)),
        ToTensor()])


def __getitem__(self, index):
    img = Image.open(self.data[index]).convert('RGB')
    target = Image.open(self.data_labels[index])

    seed = np.random.randint(1000000)  # make a seed with numpy generator
    random.seed(seed)  # apply this seed to img tranfsorms
    if self.transform is not None:
        img = self.transform(img)

    random.seed(seed)  # apply this seed to target tranfsorms
    if self.target_transform is not None:
        target = self.target_transform(target)

    target = torch.ByteTensor(np.array(target))

    return img, target

# RandomAffine, RandomApply, RandomChoice, RandomCrop, RandomHorizontalFlip, RandomOrder, RandomPerspective,
# RandomResizedCrop, RandomRotation


# RandomAffine RandomRotation RandomHorizontalFlip ColorJitter
