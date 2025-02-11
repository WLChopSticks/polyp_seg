# Import stuff
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt


# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='constant').reshape(shape)

# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))

for i in range(20):
    # Load images
    im = cv2.imread("../data/CVC-912/train/images/cvc-300_100.bmp", -1)
    im_mask = cv2.imread("../data/CVC-912/train/masks/cvc-300_100.bmp", -1)

    # Draw grid lines
    draw_grid(im, 50)
    draw_grid(im_mask, 50)

    # Merge images into separete channels (shape will be (cols, rols, 2))
    im_merge = np.concatenate((im, im_mask[..., None]), axis=2)

    # First sample...

    # Apply transformation on image
    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3, im_merge.shape[1] * 0.07, im_merge.shape[1] * 0.09)

    # Split image and mask
    im_t = im_merge_t[..., 0:-1]
    im_mask_t = im_merge_t[..., -1]

    # Display result
    plt.figure(figsize=(16, 14))
    plt.subplot(221)
    b, g, r = cv2.split(im)
    img2 = cv2.merge([r, g, b])
    plt.imshow(img2)
    plt.subplot(222)
    b, g, r = cv2.split(im_t)
    img3 = cv2.merge([r, g, b])
    plt.imshow(img3)
    plt.subplot(223)
    # b,g,r = cv2.split(im_mask)
    # img2 = cv2.merge([r,g,b])
    plt.imshow(im_mask, cmap='gray')
    plt.subplot(224)
    # b,g,r = cv2.split(im_mask_t)
    # img3 = cv2.merge([r,g,b])
    plt.imshow(im_mask_t, cmap='gray')
    # plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]])
    # plt.show()
    plt.savefig('temp/%d.jpg' % i)



