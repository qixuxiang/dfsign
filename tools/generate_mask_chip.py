# -*- coding: utf-8 -*-
"""generate chip from segmentation mask
"""

import os, sys
import cv2
import json
import numpy as np
from tqdm import tqdm
from glob import glob
from operator import add
import utils
import pdb

home = os.path.expanduser('~')
root_datadir = os.path.join(home, 'data/TT100K')
src_traindir = root_datadir + '/data/train'
src_testdir = root_datadir + '/data/test'
src_annotation = root_datadir + '/data/annotations.json'

# TT100K imageset list files
train_ids = src_traindir + '/ids.txt'
test_ids = src_testdir + '/ids.txt'

dest_datadir = root_datadir + '/TT100K_chip_voc'
image_dir = dest_datadir + '/JPEGImages'
list_dir = dest_datadir + '/ImageSets/Main'
anno_dir = dest_datadir + '/Annotations'

mask_path = os.path.join(home,
            'codes/gluon-cv/projects/seg/outdir')
            # 'codes/deeplab-tensorflow/deeplab/datasets/tt100k/exp/vis/raw_segmentation_results')

if not os.path.exists(dest_datadir):
    os.mkdir(dest_datadir)
    os.mkdir(image_dir)
    os.makedirs(list_dir)
    os.mkdir(anno_dir)


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
            chip_size = max(box_w, box_h)+100
        elif box_w < 150 and box_h < 150:
            chip_size = max(box_w, box_h)+50
        elif box_w < 200 and box_h < 200:
            chip_size = max(box_w, box_h)+100
        elif box_w < 300 and box_h < 300:
            chip_size = max(box_w, box_h)+50
        else:
            chip_size = max(box_w, box_h)

        chip = [box_cx - chip_size / 2, box_cy - chip_size / 2,
                box_cx + chip_size / 2, box_cy + chip_size / 2]

        shift_x = max(0, 0 - chip[0]) + min(0, width-1 - chip[2])
        shift_y = max(0, 0 - chip[1]) + min(0, height-1 - chip[3])

        chip = list(map(add, chip, [shift_x, shift_y]*2))
        chip_list.append([int(x) for x in chip])
    return chip_list

def main():
    with open(test_ids, 'r') as f:
        test_list = [x.strip() for x in f.readlines()]

    chip_loc = {}
    chip_name_list = []
    for imgid in tqdm(test_list):
        origin_img = cv2.imread(os.path.join(src_testdir, '%s.jpg'%imgid))
        mask_img = cv2.imread(os.path.join(mask_path, '%s.png'%imgid), cv2.IMREAD_GRAYSCALE)

        # mask_img = cv2.resize(mask_img, (2048, 2048), cv2.INTER_MAX)
        height, width = mask_img.shape[:2]
        # pdb.set_trace()
        mask_box = utils.generate_box_from_mask(mask_img)
        mask_box = list(map(utils.resize_box, mask_box,
                        [width]*len(mask_box), [2048]*len(mask_box)))
        # mask_box = utils.enlarge_box(mask_box, (2048, 2048), ratio=1)

        chip_list = mask_chip(mask_box, (2048, 2048))
        # utils._boxvis(cv2.resize(mask_img, (2048, 2048)), chip_list, origin_img)
        # cv2.waitKey(0)

        for i, chip in enumerate(chip_list):
            chip_img = origin_img[chip[1]:chip[3], chip[0]:chip[2], :].copy()
            chip_img = cv2.resize(chip_img, (416, 416), cv2.INTER_AREA)
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
