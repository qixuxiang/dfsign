import os
import random
import cv2
import numpy as np
from glob import glob
import torch
import matplotlib.pyplot as plt
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from .datasets import LoadImagesAndLabels

class VOCDetection(LoadImagesAndLabels):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root=os.path.join('~', 'data', 'voc', 'VOCdevkit'),
                splits=((2007, 'trainval'), (2012, 'trainval')), img_size=608, mode='train'):
        self._root = os.path.expanduser(root)
        self._splits = splits
        self._items = self._load_items(splits)
        self._anno_path = os.path.join('{}', 'Annotations', '{}.xml')
        self._image_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self._img_files = [self._image_path.format(*x) for x in self._items]
        self._label_files = [self._anno_path.format(*x) for x in self._items]
        self.index_map = dict(zip(self.classes, range(self.num_class)))

        super(VOCDetection, self).__init__(self._img_files, self._label_files, img_size, mode)


    @property
    def classes(self):
        """Category names."""
        return self.CLASSES

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.classes)

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for year, name in splits:
            root = os.path.join(self._root, 'VOC' + str(year))
            lf = os.path.join(root, 'ImageSets', 'Main', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _load_label(self, anno_path):
        """Parse xml file and return labels."""
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        label = []
        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text)
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append([cls_id, xmin, ymin, xmax, ymax])
        return np.array(label)
    
    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)
