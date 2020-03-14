import numpy as np
import pydensecrf.densecrf as dcrf
import os
import cv2
from PIL import Image


def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs + 1e-6)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    U = U.astype(np.float32)
    d.setUnaryEnergy(U)  # Unary

    d.addPairwiseGaussian(sxy=10, compat=3)  #
    d.addPairwiseBilateral(sxy=40, srgb=13, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q



# for (root, dirs, files) in os.walk("../data/CVC-912/train/images"):
#     print(root)
#     print(dirs)
#     print(files)
#     for file in files:
#         img_path = os.path.join(root,file)
#         pre_path = '/'.join(root.split('/')[:-1])
#         img = cv2.imread(img_path)
#         if '612' in file:
#             file = file.split('.')[0] + '.tif'
#         mask_path = os.path.join(pre_path,'masks_back',file)
#         mask=  Image.open(mask_path)
#         if mask.mode != 'L':
#             mask = mask.convert('L')
#         mask = np.array(mask)
#         out = dense_crf(img,mask)
#         save_path_pre = os.path.join(pre_path,'out_10_40_13')
#         if not os.path.exists(save_path_pre):
#             os.mkdir(save_path_pre)
#         save_path = os.path.join(save_path_pre, file)
#         out = out * 255
#         cv2.imwrite(save_path,out)
#         # cv2.imshow('img',out)
#         print(save_path)

