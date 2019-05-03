# -*- coding: utf-8 -*-
"""eval tt100k detections
"""

import os
import sys
import json
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
loc_json = os.path.join(anno_dir, 'val_chip.json')
# detections
detect_json = os.path.join(home, 'working/dfsign/yolov3/output/results.json')

def get_box_label(label_df, im_name):
    boxes = []
    labels = []
    im_label = label_df[label_df.filename == im_name]
    for index, row in im_label.iterrows():
        xmin = min(row['X1'], row['X2'], row['X3'], row['X4'])
        ymin = min(row['Y1'], row['Y2'], row['Y3'], row['Y4'])
        xmax = max(row['X1'], row['X2'], row['X3'], row['X4'])
        ymax = max(row['Y1'], row['Y2'], row['Y3'], row['Y4'])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(row['type'])
    return np.array(boxes) - 1, labels


def result_eval(detections, label_df):
    pred_num = []
    label_num = []
    tp_num = []
    for img_id, preds in detections.items():
        gt_boxes, gt_types = get_box_label(label_df, img_id+'.jpg')
        pred_num.append(len(preds))
        label_num.append(len(gt_boxes))
        correct = 0
        for pred in preds:
            pred_box = pred[:4]
            pred_type = pred[4]
            for gt_box, gt_type in zip(gt_boxes, gt_types):
                if gt_type == pred_type and utils.overlap(pred_box, gt_box, 0.5):
                    correct += 1
                    break
        tp_num.append(correct)

    recall = 1.0 * sum(tp_num) / sum(label_num)
    precision = 1.0 * sum(tp_num) / sum(pred_num)
    return recall, precision

def main():
    # read chip loc
    with open(loc_json, 'r') as f:
        chip_loc = json.load(f)
    # read chip detections
    with open(detect_json, 'r') as f:
        chip_detect = json.load(f)

    # read annotations
    df = pd.read_csv(src_annotation)

    dfsign_results = {}
    for chip_id, chip_result in chip_detect.items():
        img_id = chip_id.split('_')[0]

        img_result = []
        loc = chip_loc[chip_id]['loc']
        for i, pred_box in enumerate(chip_result['pred_box']):
            # transform to orginal image
            ratio = (loc[2] - loc[0]) / 416.
            pred_box = [pred_box[0] * ratio + loc[0],
                        pred_box[1] * ratio + loc[1],
                        pred_box[2] * ratio + loc[0],
                        pred_box[3] * ratio + loc[1]]
            sign_type = int(chip_result['pred_label'][i])
            score = chip_result['pred_score'][i]
            img_result.append(pred_box + [sign_type, score])
        if img_id in dfsign_results:
            dfsign_results[img_id] += img_result
        else:
            dfsign_results[img_id] = img_result

    recall, precision = result_eval(dfsign_results, df)
    F1 = 2.0 / (1 / recall + 1 / precision)
    print('recall:%.3f, precision:%.3f, F1:%.3f' % (recall, precision, F1))

if __name__ == '__main__':
    main()
