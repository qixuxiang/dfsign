import os
import sys
import shutil
import pickle
import numpy as np

import pdb

home = os.path.expanduser('~')
root_datadir = os.path.join(home, 'data/dfsign')

dest_datadir = root_datadir + '/dfsign_chip_voc'
image_dir = dest_datadir + '/JPEGImages'
list_dir = dest_datadir + '/ImageSets/Main'
anno_dir = dest_datadir + '/Annotations'

sample_dir = '../mmdetection/dfsign/samples'

with open('miss_images.pkl', 'rb') as f:
    missing = pickle.load(f)
    missing = [x[:-4] for x in missing]
with open(os.path.join(list_dir, 'test.txt'), 'r') as f:
    all_images = [x.strip() for x in f.readlines()]

missing_images = []
for x in missing:
    for y in all_images:
        if x in y:
            missing_images.append(y)

missing_images = [os.path.join(image_dir, x+'.jpg') for x in missing_images]
if os.path.exists(sample_dir):
    shutil.rmtree(sample_dir)
os.mkdir(sample_dir)
for x in missing_images:
    shutil.copy(x, sample_dir)