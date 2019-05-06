import sys, os
import cv2
import glob
import numpy as np

userhome = os.path.expanduser('~')

mask_path = os.path.join(userhome, 'working/dfsign/pytorch-deeplab-xception/run/mask')

mask_list = glob.glob(mask_path + '/*.png')
for mask in mask_list:
    im = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    if im.sum() == 0:
        print(mask)
        