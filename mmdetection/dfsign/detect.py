import argparse

import os.path as osp
import os
import sys
import glob
import json
import time
import numpy as np
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.datasets import DFSignDataset
from mmdet.models import build_detector, detectors
from mmdet.apis import inference_detector

import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def show_result(img, result, classes, score_thr=0.3, out_file=None):
    img = mmcv.imread(img)
    class_names = classes
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None)


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    
    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint)
    model = MMDataParallel(model, device_ids=[0])

    dfsign = True
    # get image list
    if dfsign:
        root_dir = '../data/dfsign/dfsign_chip_voc'
        root_dir = os.path.expanduser(root_dir)
        list_file = os.path.join(root_dir, 'ImageSets/Main/test.txt')
        image_dir = os.path.join(root_dir, 'JPEGImages')
        with open(list_file, 'r') as f:
            images = [x.strip() for x in f.readlines()]
        imglist = [os.path.join(image_dir, x+'.jpg') for x in images]
    else:
        imglist = glob.glob('samples/*.jpg')

    classes = DFSignDataset.CLASSES
    results = {}
    t0 = time.time()
    for i, preds in enumerate(inference_detector(model, imglist, cfg, device='cuda:0')):
        detect_time = time.time() - t0
        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s   \r'
                            .format(i + 1, len(imglist), detect_time))
        sys.stdout.flush()

        if args.show:
            show_result(imglist[i], preds, classes)
        else:
            img_id = imglist[i][:-4]
            box = np.vstack(preds)
            if box.shape[0] == 0:
                continue
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(preds)
            ]
            labels = np.concatenate(labels)
            results[img_id] = {'pred_box': box[:, :4],
                            'pred_score': box[:, 4],
                            'pred_label': [classes[i] for i in labels]}
        t0 = time.time()

    if not args.show:
        with open('results.json', 'w') as f:
            json.dump(results, f, cls=MyEncoder)
            print('results json saved.')


if __name__ == '__main__':
    main()
