import os
from PIL import Image
import cv2

out_dir = '../data/CVC-912/test/images'
gt_dir = ''

for (root, dirs, files) in os.walk(out_dir):
    for file in files:
        image_path = os.path.join(out_dir, file)
        image = cv2.imread(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (384, 288))
        cv2.imwrite(image_path, image)



