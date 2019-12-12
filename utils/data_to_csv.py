import os
import pandas as pd


images_root = '/home/jiaxin/MICCAI2020/data/CVC-912-fixed/train/images'
masks_root = '/home/jiaxin/MICCAI2020/data/CVC-912-fixed/train/masks'

images = os.listdir(images_root)
masks = []

for i in images:
    if i.find('cvc-612') >= 0:
        masks.append(i.split('.')[0] + '.tif')
    elif i.find('cvc-300') >= 0:
        masks.append(i)
    else:
        RuntimeError("")

dataframe = pd.DataFrame([images, masks])
dataframe = dataframe.transpose()
dataframe.to_csv('/home/jiaxin/MICCAI2020/data/CVC-912-fixed/csv/train.csv', index=False, header=['image', 'mask'])
