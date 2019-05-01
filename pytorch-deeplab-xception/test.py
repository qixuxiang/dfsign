import argparse
import os
import time
import cv2
import shutil
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.deeplab import *
from dataloaders.datasets import tt100k
from torch.utils.data import DataLoader
import pdb

class Tester(object):
    def __init__(self, args):
        self.args = args
        
        # Define Dataloader
        test_set = tt100k.TT100KSegmentation(args, split='test')
        self.nclass = test_set.NUM_CLASSES
        self.test_loader = DataLoader(test_set,
                                batch_size=args.test_batch_size,
                                shuffle=False,
                                num_workers=args.workers)

        # Define network
        self.model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride)

        # Using cuda
        if args.cuda:
            torch.cuda.set_device(self.args.gpu_ids)
            self.model = self.model.cuda()

        # load weight
        assert args.weight is not None
        if not os.path.isfile(args.weight):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.weight))
        checkpoint = torch.load(args.weight)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.weight))

        self.show = False
        self.outdir = 'run/mask'
        if not self.show:
            if os.path.exists(self.outdir):
                shutil.rmtree(self.outdir)
            os.makedirs(self.outdir)

    def test(self):
        self.model.eval()
        t0 = time.time()
        for i, sample in enumerate(self.test_loader):
            images, targets, paths = sample['image'], sample['label'], sample['path']
            if self.args.cuda:
                images, target = images.cuda(), targets.cuda()
            with torch.no_grad():
                output = self.model(images)
            preds = output.data.cpu().numpy()
            targets = targets.cpu().numpy()
            preds = np.argmax(preds, axis=1)
            print('batch %d, time:%f' % (i, time.time() - t0))

            if self.show:
                for path, pred in zip(paths, preds):
                    self.show_result(path, pred)

            else:
                for path, pred in zip(paths, preds):
                    self.save_mask(self.outdir, pred, path)

            t0 = time.time()

    def show_result(self, path, pred):
        import matplotlib.pyplot as plt
        img = cv2.imread(path)
        img = cv2.resize(img, (self.args.crop_size, self.args.crop_size))
        
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1).imshow(img[:, :, ::-1])
        plt.subplot(1, 2, 2).imshow(pred.astype(np.uint8) * 255)
        plt.show()
        cv2.waitKey()
    
    def save_mask(self, outdir, pred, path):
        name = path.split('/')[-1][:-4] + '.png'
        cv2.imwrite(os.path.join(outdir, name), pred.astype(np.uint8))

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'tt100k'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=1,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=640,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=640,
                        help='crop image size')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--weight', type=str, default=None,
                        help='put the path to resuming file if needed')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            assert len(args.gpu_ids) == 1
            args.gpu_ids = args.gpu_ids[0]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    print(args)
    torch.manual_seed(args.seed)
    tester = Tester(args)
    tester.test()
    
if __name__ == "__main__":
   main()
