# -*- coding: utf-8 -*-
"""submit
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
import utils

import pdb

home = os.path.expanduser('~')
root_datadir = os.path.join(home, 'data/dfsign')
src_traindir = root_datadir + '/train'
src_testdir = root_datadir + '/test'
src_annotation = root_datadir + '/train_label_fix.csv'

dest_datadir = root_datadir + '/dfsign_chip_voc'
image_dir = dest_datadir + '/JPEGImages'
list_dir = dest_datadir + '/ImageSets/Main'
anno_dir = dest_datadir + '/Annotations'

# chip loc
loc_json = os.path.join(anno_dir, 'test_chip.json')
# detections
detect_json = os.path.join(home, 'working/dfsign/mmdetection/dfsign/results.json')

def main():
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
            pred_box = [pred_box[0] + loc[0] + 1,
                        pred_box[1] + loc[1] + 1,
                        pred_box[2] + loc[0] + 1,
                        pred_box[3] + loc[1] + 1]
            sign_type = int(chip_result['pred_label'][i])
            score = chip_result['pred_score'][i]
            pred_box = [pred_box[0], pred_box[1], pred_box[2], pred_box[1],
                        pred_box[2], pred_box[3], pred_box[0], pred_box[3]]
            # pred_box = [int(x) for x in pred_box]
            dfsign_results.append([img_id] + pred_box + [sign_type, score])
    
    filter_results = []
    temp = np.array(dfsign_results)
    detected_img = np.unique(temp[:, 0])
    for img_id in detected_img:
        preds = temp[temp[:, 0] == img_id]
        preds = preds[preds[:, -1].argsort()]
        filter_results.append(list(preds[-1])[:-1])

    test_list = glob.glob(src_testdir + '/*.jpg')
    test_list = [os.path.basename(x) for x in test_list]
    addition = []
    temp = np.array(filter_results)
    for img_id in test_list:
        if img_id not in temp[:, 0]:
            addition.append([img_id] + [0]*9)
    
    filter_results += addition
    columns = ['filename','X1','Y1','X2','Y2','X3','Y3','X4','Y4','type']
    df = pd.DataFrame(filter_results, columns=columns)
    df.to_csv('predict.csv', index=False)

    

if __name__ == '__main__':
    main()
