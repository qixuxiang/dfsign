# -*- coding: utf-8 -*-
"""eval tt100k detections
"""

import os
import sys
import json
import numpy as np
import utils

import pdb

home = os.path.expanduser('~')
root_datadir = os.path.join(home, 'data/TT100K')
src_traindir = root_datadir + '/data/train'
src_testdir = root_datadir + '/data/test'
src_annotation = root_datadir + '/data/annotations.json'

dest_datadir = root_datadir + '/TT100K_chip_voc'
image_dir = dest_datadir + '/JPEGImages'
list_dir = dest_datadir + '/ImageSets/Main'
anno_dir = dest_datadir + '/Annotations'

# chip loc
loc_json = os.path.join(anno_dir, 'test_chip.json')
# detections
detect_json = os.path.join(home, 'codes/gluon-cv/projects/yolo/results/results.json')

# add path
sys.path.append(os.path.join(root_datadir, 'code/python'))
import anno_func

def main():
    # read chip loc
    with open(loc_json, 'r') as f:
        chip_loc = json.load(f)
    # read chip detections
    with open(detect_json, 'r') as f:
        chip_detect = json.load(f)
    # read tt100k test set annotations
    annos = json.loads(open(src_annotation).read())

    tt100k_results = {}
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
            img_result.append({'category': chip_result['pred_label'][i],
                               'score': chip_result['pred_score'][i]*1000,
                               'bbox': {'xmin': pred_box[0],
                                        'ymin': pred_box[1],
                                        'xmax': pred_box[2],
                                        'ymax': pred_box[3]}})
        if img_id in tt100k_results:
            tt100k_results[img_id]['objects'] += img_result
        else:
            tt100k_results[img_id] = {'objects': img_result}
        # break

    results_annos = {'imgs': tt100k_results}

    # print(results_annos)
    # pdb.set_trace()
    
    sm = anno_func.eval_annos(annos, results_annos, iou=0.5, check_type=True, types=anno_func.type45,
                         minboxsize=0,maxboxsize=400)
    print(sm['report'])
    sm = anno_func.eval_annos(annos, results_annos, iou=0.5, check_type=True, types=anno_func.type45,
                            minboxsize=0,maxboxsize=32)
    print(sm['report'])
    sm = anno_func.eval_annos(annos, results_annos, iou=0.5, check_type=True, types=anno_func.type45,
                            minboxsize=32,maxboxsize=96)
    print(sm['report'])
    sm = anno_func.eval_annos(annos, results_annos, iou=0.5, check_type=True, types=anno_func.type45,
                            minboxsize=96,maxboxsize=400)
    print(sm['report'])
    
    with open('result_annos.json', 'w') as f:
        json.dump(results_annos, f)

if __name__ == '__main__':
    main()
