# -*- coding: utf-8 -*-
"""generate chip from segmentation mask
"""

import os, sys
import cv2
import glob
import json
import random
import numpy as np
from tqdm import tqdm
from operator import add
import utils
import pdb

random.seed(100)
home = os.path.expanduser('~')
root_datadir = os.path.join(home, 'data/dfsign')
src_traindir = root_datadir + '/train'
src_testdir = root_datadir + '/test'
src_annotation = root_datadir + '/train_label_fix.csv'

dest_datadir = root_datadir + '/dfsign_chip_voc'
image_dir = dest_datadir + '/JPEGImages'
list_dir = dest_datadir + '/ImageSets/Main'
anno_dir = dest_datadir + '/Annotations'

mask_path = os.path.join(home, 'working/dfsign/pytorch-deeplab-xception/run/mask')

def mask_chip(mask_box, image_size):
    """
    Args:
        mask_box: list of box, [xmin, ymin, xmax, ymax]
        image_size: (width, height)
    Returns:
        chips: list of box
    """

    width, height = image_size

    chip_list = []
    for box in mask_box:
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        box_cx = box[0] + box_w / 2
        box_cy = box[1] + box_h / 2

        if box_w < 100 and box_h < 100:
            chip_size = max(box_w, box_h)+50
        elif box_w < 150 and box_h < 150:
            chip_size = max(box_w, box_h)+50
        elif box_w < 200 and box_h < 200:
            chip_size = max(box_w, box_h)+150
        elif box_w < 300 and box_h < 300:
            chip_size = max(box_w, box_h)+200
        else:
            chip_size = max(box_w, box_h)+200

        chip = [box_cx - chip_size / 2, box_cy - chip_size / 2,
                box_cx + chip_size / 2, box_cy + chip_size / 2]

        shift_x = max(0, 0 - chip[0]) + min(0, width-1 - chip[2])
        shift_y = max(0, 0 - chip[1]) + min(0, height-1 - chip[3])

        chip = list(map(add, chip, [shift_x, shift_y]*2))
        chip_list.append([int(x) for x in chip])
    return chip_list

def main():
    if not os.path.exists(dest_datadir):
        os.mkdir(dest_datadir)
        os.mkdir(image_dir)
        os.makedirs(list_dir)
        os.mkdir(anno_dir)

    train_list = glob.glob(src_traindir + '/*.jpg')
    random.shuffle(train_list)
    train_list, val_list = train_list[:-2000], train_list[-2000:]
    test_list = glob.glob(src_testdir + '/*.jpg')
    test_list = [os.path.basename(x)[:-4] for x in test_list]

    chip_loc = {}
    chip_name_list = []
    for imgid in tqdm(test_list):
        origin_img = cv2.imread(os.path.join(src_testdir, '%s.jpg'%imgid))
        mask_img = cv2.imread(os.path.join(mask_path, '%s.png'%imgid), cv2.IMREAD_GRAYSCALE)
        
        height, width = mask_img.shape[:2]
        mask_box = utils.generate_box_from_mask(mask_img)
        mask_box = list(map(utils.resize_box, mask_box,
                            [(width, height)]*len(mask_box), 
                            [(3200, 1800)]*len(mask_box)))

        chip_list = mask_chip(mask_box, (3200, 1800))
        # utils._boxvis(cv2.resize(mask_img, (2048, 2048)), chip_list, origin_img)
        # cv2.waitKey(0)

        for i, chip in enumerate(chip_list):
            chip_img = origin_img[chip[1]:chip[3], chip[0]:chip[2], :].copy()
            # chip_img = cv2.resize(chip_img, (416, 416), cv2.INTER_AREA)
            chip_name = '%s_%d' % (imgid, i)
            cv2.imwrite(os.path.join(image_dir, '%s.jpg'%chip_name), chip_img)
            chip_name_list.append(chip_name)

            chip_info = {'loc': chip}
            chip_loc[chip_name] = chip_info

    # write test txt
    with open(os.path.join(list_dir, 'test.txt'), 'w') as f:
        f.writelines([x+'\n' for x in chip_name_list])
        print('write txt.')

    # write chip loc json
    with open(os.path.join(anno_dir, 'test_chip.json'), 'w') as f:
        json.dump(chip_loc, f)
        print('write json.')

if __name__ == '__main__':
    main()
