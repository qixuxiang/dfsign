import os
import numpy as np
import json
import utils.tt100k_utils as utils

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.label_object = []
        self.detect_object = []
        self.mask_object = []
        user_home = os.path.expanduser('~')
        datadir = os.path.join(user_home, 'data/TT100K')
        self.tt100k_annos = json.loads(open(datadir + '/data/annotations.json').read())

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Region_Recall(self):
        return np.sum(self.detect_object) / np.sum(self.label_object)
    
    def Region_Num(self):
        return np.mean(self.mask_object)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def _generate_count(self, pre_image, paths):
        for mask_img, path in zip(pre_image, paths):
            imgid = str(path.split('/')[-1][:-4])
            label_box = utils.get_label_box(self.tt100k_annos, imgid)

            height, width = mask_img.shape[:2]
            mask_box = utils.generate_box_from_mask(mask_img.astype(np.uint8))
            mask_box = list(map(utils.resize_box, mask_box,
                            [width]*len(mask_box), [2048]*len(mask_box)))
            mask_box = utils.enlarge_box(mask_box, (2048, 2048), ratio=1.3)

            count = 0
            for box1 in label_box:
                for box2 in mask_box:
                    if utils.overlap(box2, box1):
                        count += 1
                        break
            self.label_object.append(len(label_box))
            self.detect_object.append(count)
            self.mask_object.append(len(mask_box))

    def add_batch(self, gt_image, pre_image, paths):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self._generate_count(pre_image, paths)
        

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.label_object = []
        self.detect_object = []
        self.mask_object = []

    