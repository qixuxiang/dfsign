"""Faster RCNN Demo script."""
import os,sys
import glob
import argparse
import mxnet as mx
import gluoncv as gcv
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Test with Faster RCNN networks.')
    parser.add_argument('--network', type=str, default='faster_rcnn_resnet50_v1b_coco',
                        help="Faster RCNN full network name")
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters. You can specify parameter file name.')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Threshold of object score when visualize the bboxes.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    dfsign = False
    if dfsign:
        root_dir = '~/data/dfsign/dfsign_chip_voc'
        root_dir = os.path.expanduser(root_dir)
        list_file = os.path.join(root_dir, 'ImageSets/Main/test.txt')
        image_dir = os.path.join(root_dir, 'JPEGImages')
        with open(list_file, 'r') as f:
            images = [x.strip() for x in f.readlines()]
        image_list = [os.path.join(image_dir, x+'.jpg') for x in images]
    else:
        image_list = glob.glob('samples/*.jpg')

    kwargs = {}
    module_list = []
    if args.norm_layer is not None:
        module_list.append(args.norm_layer)
        if args.norm_layer == 'bn':
            kwargs['num_devices'] = len(args.gpus.split(','))
    kwargs['classes'] = gdata.DFSignDetection.CLASSES
    net_name = '_'.join(('faster_rcnn', *module_list, 'resnet101_v1d', 'custom'))
    net = gcv.model_zoo.get_model(net_name, pretrained=False, pretrained_base=False, **kwargs)
    net.load_parameters(args.pretrained)
    net.set_nms(0.3, nms_topk=200, post_nms=5)
    net.collect_params().reset_ctx(ctx = ctx)

    for image in image_list:
        ax = None
        x, img = presets.rcnn.load_test(image, short=net.short, max_size=net.max_size)
        x = x.as_in_context(ctx[0])
        ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
        ax = gcv.utils.viz.plot_bbox(img, bboxes, scores, ids, thresh=args.thresh,
                                     class_names=net.classes, ax=ax)
        plt.show()
