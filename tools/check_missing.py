import os
import sys
import json
import shutil
import pickle
import numpy as np

import pdb

home = os.path.expanduser('~')
root_datadir = os.path.join(home, 'data/dfsign')

dest_datadir = root_datadir + '/dfsign_detect_voc'
image_dir = dest_datadir + '/JPEGImages'
list_dir = dest_datadir + '/ImageSets/Main'
anno_dir = dest_datadir + '/Annotations'

sample_dir = '../mmdetection/dfsign/samples'

img_big = []
with open(os.path.join(anno_dir, 'test_chip.json'), 'r') as f:
    chip_loc = json.load(f)
for name, chip in chip_loc.items():
    chip = chip['loc']
    width = chip[2] - chip[0]
    height = chip[3] - chip[1]
    if max(width, height) > 500:
        img_big.append(name)
print(len(img_big))

check_big = True
check_missing = False
img_list = []
if check_missing:
    with open('miss_images.pkl', 'rb') as f:
        missing = pickle.load(f)
        missing = [x[:-4] for x in missing]
    with open(os.path.join(list_dir, 'test.txt'), 'r') as f:
        all_images = [x.strip() for x in f.readlines()]

    for x in missing:
        for y in all_images:
            if x in y:
                img_list.append(y)
elif check_big:
    img_list = img_big

img_list = [os.path.join(image_dir, x+'.jpg') for x in img_list]
if os.path.exists(sample_dir):
    shutil.rmtree(sample_dir)
os.mkdir(sample_dir)
for x in img_list:
    shutil.copy(x, sample_dir)