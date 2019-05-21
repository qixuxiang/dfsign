# -*- coding: utf-8 -*-

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

old_datadir = root_datadir + '/dfsign_chip_voc'
old_anno_dir = old_datadir + '/Annotations'

dest_datadir = root_datadir + '/dfsign_detect_voc'
image_dir = dest_datadir + '/JPEGImages'
list_dir = dest_datadir + '/ImageSets/Main'
anno_dir = dest_datadir + '/Annotations'

loc_json = os.path.join(old_anno_dir, 'test_chip.json')
detect_json = os.path.join(home,
                        'working/dfsign/mmdetection/dfsign/results_chip.json')

def main():
    if not os.path.exists(dest_datadir):
        os.mkdir(dest_datadir)
        os.mkdir(image_dir)
        os.makedirs(list_dir)
        os.mkdir(anno_dir)

    # read chip loc
    with open(loc_json, 'r') as f:
        chip_loc = json.load(f)
    # read chip detections
    with open(detect_json, 'r') as f:
        chip_detect = json.load(f)

    dfsign_results = []
    for chip_id, chip_result in chip_detect.items():
        chip_id = os.path.basename(chip_id)
        img_id = chip_id.split('_')[0] + '.jpg'

        loc = chip_loc[chip_id]['loc']
        for i, pred_box in enumerate(chip_result['pred_box']):
            # transform to orginal image
            # ratio = (loc[2] - loc[0]) / 416.
            pred_box = [pred_box[0] + loc[0],
                        pred_box[1] + loc[1],
                        pred_box[2] + loc[0],
                        pred_box[3] + loc[1]]
            sign_type = int(chip_result['pred_label'][i])
            score = chip_result['pred_score'][i]
            dfsign_results.append([img_id] + pred_box + [sign_type, score])
    
    filter_results = []
    temp = np.array(dfsign_results)
    detected_img = np.unique(temp[:, 0])
    for img_id in detected_img:
        preds = temp[temp[:, 0] == img_id]
        preds = preds[preds[:, -1].argsort()]
        filter_results.append(list(preds[-1])[:-1])

    chip_loc = {}
    chip_name_list = []
    for result in tqdm(filter_results):
        imgid = result[0][:-4]

        box = [float(x) for x in result[1:5]]
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]

        if max(box_w, box_h) < 30:
            ratio = 5
        else:
            ratio = 3.5

        region_w = max(box_w, box_h) * ratio
        region_h = region_w
        center_x = box[0] + box_w / 2.0
        center_y = box[1] + box_h / 2.0
        region = [center_x - region_w / 2, 
                  center_y - region_h / 2,
                  center_x + region_w / 2,
                  center_y + region_h / 2]
        shift_x = max(0, 0 - region[0]) + min(0, 3200 - 1 - region[2])
        shift_y = max(0, 0 - region[1]) + min(0, 1800 - 1 - region[3])
        chip = [region[0] + shift_x,
                region[1] + shift_y,
                region[2] + shift_x,
                region[3] + shift_y]
        chip = [int(x) for x in chip]

        origin_img = cv2.imread(os.path.join(src_testdir, '%s.jpg'%imgid))
        chip_img = origin_img[chip[1]:chip[3], chip[0]:chip[2], :].copy()
        chip_name = '%s_%d' % (imgid, 0)
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
