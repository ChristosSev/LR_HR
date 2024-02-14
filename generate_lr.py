import os
import cv2
import numpy as np


def modcrop(imgs, modulo):
    if len(imgs.shape) == 2:
        sz = imgs.shape
        sz = sz - np.mod(sz, modulo)
        imgs = imgs[0:sz[0], 0:sz[1]]
    else:
        tmpsz = imgs.shape
        sz = tmpsz[0:2]
        sz = sz - np.mod(sz, modulo)
        imgs = imgs[0:sz[0], 0:sz[1], :]
    return imgs

folder = './Dataset/test_data/'
scale = 4

# Generate data
filepaths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg') or f.endswith('.tiff')]
for i, filepath in enumerate(filepaths):
    image = cv2.imread(filepath)
    im_label = modcrop(image, scale)
    im_lr = cv2.resize(im_label, (im_label.shape[1] // scale, im_label.shape[0] // scale), interpolation=cv2.INTER_CUBIC)
    os.makedirs('./lr_images_train', exist_ok=True)
    os.makedirs('./hr_images_train', exist_ok=True)
    cv2.imwrite(os.path.join('./lr_images_train', f'{i}.png'), im_lr)
    cv2.imwrite(os.path.join('./hr_images_train', f'{i}_gt.png'), im_label)
