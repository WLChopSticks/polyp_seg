import cv2
from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import os

image_dir = '/Users/wanglei/Documents/polyp_seg/polyp_seg/data/CVC-912/tr/images3'

for (root, dirs, files) in os.walk(image_dir):
    for file in files:
        image_path = os.path.join(image_dir, file)
        img = cv2.imread(image_path)

        segments = slic(img, n_segments=100, compactness=10)
        out = mark_boundaries(img, segments)

        cv2.imwrite(image_path, out * 255)
        print(image_path)
