from __future__ import division
from __future__ import print_function

import sys, os
sys.path.append('../..')

import argparse
import time
import json
import cv2
import numpy as np
import mxnet as mx
from tqdm import tqdm
import matplotlib.pyplot as plt
from mxnet import nd
from mxnet import gluon
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

import pdb

userhome = os.path.expanduser('~')

def parse_args():
    parser = argparse.ArgumentParser(description='Eval SSD networks.')
    parser.add_argument('--network', type=str, default='vgg16_atrous',
                        help="Base network name")
    parser.add_argument('--data-shape', type=int, default=300,
                        help="Input data shape")
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset.')
    parser.add_argument('--dataset_root', type=str, default=os.path.join(userhome, 'data/VOCdevkit'),
                        help='Training dataset root.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--save-dir', dest='save_dir', type=str, default='results',
                        help='Saving parameter dir')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
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

def get_dataset(dataset, data_shape):
    if dataset.lower() == 'voc':
        val_dataset = gdata.VOCDetection(root=args.dataset_root, splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(data_shape, data_shape))
    elif dataset.lower() == 'tt100k':
        val_dataset = gdata.TT100KDetection(root=args.dataset_root, splits='test', preload_label=False)
        val_metric = None
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric

def _visdetection(image, detections, labels):
    detections = np.array(detections).astype(np.int32)
    for box, label in zip(detections, labels):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), 100, 2)
        cv2.putText(image, label, (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    plt.imshow(image)
    plt.show()

def test(net, val_dataset, ctx, classes, size):
    """Test on test dataset."""
    items = val_dataset._items
    results = dict()

    net.collect_params().reset_ctx(ctx)
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()
    for idx in tqdm(range(size)):
        im_id = items[idx][1]
        im_fname = os.path.join('{}', 'JPEGImages', '{}.jpg').format(*items[idx])
        x, img = gdata.transforms.presets.ssd.load_test(im_fname, short=300)
        ids, scores, bboxes = net(x.copyto(ctx[0]))

        ids = ids.astype('int32').asnumpy().squeeze()
        bboxes = bboxes.asnumpy().squeeze()
        scores = scores.asnumpy().squeeze()
        mask = ids > -1
        # pdb.set_trace()
        # _visdetection(img, bboxes, [classes[i] for i in ids[mask]])
        # cv2.waitKey(0)
        results[im_id] = {'pred_box': bboxes[mask],
                           'pred_score': scores[mask],
                           'pred_label': [classes[i] for i in ids[mask]]}
    return results

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # test contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    if args.dataset == 'tt100k':
        net_name = '_'.join(('ssd', str(args.data_shape), args.network, 'custom'))
        args.save_prefix += 'tt100k'
        net = gcv.model_zoo.get_model(net_name, classes=gdata.TT100KDetection.CLASSES,
                                      pretrained=False, pretrained_base=False)
        net.load_parameters(args.pretrained.strip())
    else:
        net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.dataset))
        args.save_prefix += net_name
        if args.pretrained.lower() in ['true', '1', 'yes', 't']:
            net = gcv.model_zoo.get_model(net_name, pretrained=True)
        else:
            net = gcv.model_zoo.get_model(net_name, pretrained=False)
            net.load_parameters(args.pretrained.strip())

    # test data
    val_dataset, val_metric = get_dataset(args.dataset, args.data_shape)
    classes = val_dataset.classes  # class names

    # testing
    results = test(net, val_dataset, ctx, classes, len(val_dataset))
    with open(args.save_dir + '/results.json', 'w') as f:
        json.dump(results, f, cls=MyEncoder)
