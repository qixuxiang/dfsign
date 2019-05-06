import os, sys
import cv2
import time
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt
sys.path.append('../..')

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.data import get_segmentation_dataset, ms_batchify_fn
from gluoncv.utils.viz import get_color_pallete
from gluoncv.utils.parallel import *

from deeplabv3 import get_deeplab
from region_net import get_regionnet
import pdb

userhome = os.path.expanduser('~')

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone name (default: resnet50)')
    parser.add_argument('--dataset', type=str, default='pascalaug',
                        help='dataset name (default: pascal)')
    parser.add_argument('--dataset_root', type=str, default=os.path.join(userhome, 'data/VOCdevkit'),
                        help='Training dataset root.')
    parser.add_argument('--workers', type=int, default=16,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=480,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default= False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.5,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, \
                        and beta/gamma for batchnorm layers.')
    # cuda and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs (default: 4)')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='default',
                        help='set the checkpoint name')
    parser.add_argument('--model-zoo', type=str, default=None,
                        help='evaluating on model zoo model')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default= False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default= False,
                        help='using Synchronized Cross-GPU BatchNorm')

    # the parser
    args = parser.parse_args()
    # handle contexts
    if args.no_cuda:
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        print('Number of GPUs:', args.ngpus)
        args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    print(args)
    return args

def test(args):
    # output folder
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # image transform
    input_transform = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    if args.eval:
        testset = get_segmentation_dataset(
            args.dataset, split='val', mode='testval', transform=input_transform)
        total_inter, total_union, total_correct, total_label = \
            np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    else:
        testset = get_segmentation_dataset(args.dataset,
            root=args.dataset_root, split='test', mode='test', transform=input_transform)
    test_data = gluon.data.DataLoader(
        testset, args.test_batch_size, shuffle=False, last_batch='keep',
        batchify_fn=ms_batchify_fn, num_workers=args.workers)
    # create network
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        if 'region' in args.dataset:
            model = get_regionnet(dataset=args.dataset, ctx=args.ctx,
                                    backbone=args.backbone, norm_layer=args.norm_layer,
                                    norm_kwargs=args.norm_kwargs, aux=args.aux,
                                    crop_size=args.crop_size)
        else:
            model = get_deeplab(dataset=args.dataset, ctx=args.ctx,
                                backbone=args.backbone, norm_layer=args.norm_layer,
                                norm_kwargs=args.norm_kwargs, aux=args.aux,
                                crop_size=args.crop_size)
        # load pretrained weight
        assert args.resume is not None, '=> Please provide the checkpoint using --resume'
        if os.path.isfile(args.resume):
            model.load_parameters(args.resume, ctx=args.ctx, ignore_extra=True)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'" \
                .format(args.resume))
    # print(model)
    # evaluator = MultiEvalModel(model, testset.num_class, ctx_list=args.ctx, scales=[1.0], flip=False)
    evaluator = SegEvalModel(model)
    metric = gluoncv.utils.metrics.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)
    for i, (data, dsts) in enumerate(tbar):
        if args.eval:
            predicts = [pred[0] for pred in evaluator.parallel_forward(data)]
            targets = [target.as_in_context(predicts[0].context) \
                       for target in dsts]
            metric.update(targets, predicts)
            pixAcc, mIoU = metric.get()
            tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
        else:
            im_paths = dsts
            # pdb.set_trace()
            # predicts = evaluator.parallel_forward(data)
            predicts = evaluator(data[0].as_in_context(args.ctx[0]).expand_dims(0))
            for predict, impath in zip([predicts], im_paths):
                predict = mx.nd.squeeze(mx.nd.argmax(predict[0], 0)).asnumpy() + \
                    testset.pred_offset
                mask = predict * 255
                # mask = cv2.resize(mask, (614, 614))
                outname = os.path.splitext(impath)[0] + '.png'
                cv2.imwrite(os.path.join(outdir, outname), mask)
                # mask = get_color_pallete(predict, args.dataset)
                # outname = os.path.splitext(impath)[0] + '.png'
                # mask.save(os.path.join(outdir, outname))

if __name__ == "__main__":
    args = parse_args()
    args.test_batch_size = args.ngpus
    print('Testing model: ', args.resume)
    test(args)
