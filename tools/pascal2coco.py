"""Convert PASCAL VOC annotations to MSCOCO format and save to a json file.
The MSCOCO annotation has following structure:
{
    "images": [
        {
            "file_name": ,
            "height": ,
            "width": ,
            "id":
        },
        ...
    ],
    "type": "instances",
    "annotations": [
        {
            "segmentation": [],
            "area": ,
            "iscrowd": ,
            "image_id": ,
            "bbox": [],
            "category_id": ,
            "id": ,
            "ignore":
        },
        ...
    ],
    "categories": [
        {
            "supercategory": ,
            "id": ,
            "name":
        },
        ...
    ]
}
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import os.path as osp
from collections import OrderedDict
import json

import xmltodict
import mmcv

logger = logging.getLogger(__name__)


class PASCALVOC2COCO(object):
    """Converters that convert PASCAL VOC annotations to MSCOCO format."""

    def __init__(self):
        self.cat2id = {
            '1': 1, '2': 2, '3': 3, '4': 4,
            '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, '10': 10, '11': 11, '12': 12,
            '13': 13, '14': 14, '15': 15, '16': 16,
            '17': 17, '18': 18, '19': 19, '20': 20, '21':21,
        }

    def get_img_item(self, file_name, image_id, size):
        """Gets a image item."""
        image = OrderedDict()
        image['file_name'] = file_name
        image['height'] = int(size['height'])
        image['width'] = int(size['width'])
        image['id'] = image_id
        return image

    def get_ann_item(self, obj, image_id, ann_id):
        """Gets an annotation item."""
        x1 = int(obj['bndbox']['xmin']) - 1
        y1 = int(obj['bndbox']['ymin']) - 1
        w = int(obj['bndbox']['xmax']) - x1
        h = int(obj['bndbox']['ymax']) - y1

        annotation = OrderedDict()
        annotation['segmentation'] = [[x1, y1, x1, (y1 + h), (x1 + w), (y1 + h), (x1 + w), y1]]
        annotation['area'] = w * h
        annotation['iscrowd'] = 0
        annotation['image_id'] = image_id
        annotation['bbox'] = [x1, y1, w, h]
        annotation['category_id'] = self.cat2id[obj['name']]
        annotation['id'] = ann_id
        annotation['ignore'] = int(obj['difficult'])
        return annotation

    def get_cat_item(self, name, id):
        """Gets an category item."""
        category = OrderedDict()
        category['supercategory'] = 'none'
        category['id'] = id
        category['name'] = name
        return category

    def convert(self, devkit_path, split, save_file):
        """Converts PASCAL VOC annotations to MSCOCO format. """
        split_file = osp.join(devkit_path, 'ImageSets/Main/{}.txt'.format(split))
        ann_dir = osp.join(devkit_path, 'Annotations')

        name_list = mmcv.list_from_file(split_file)

        images, annotations = [], []
        ann_id = 1
        for id, name in enumerate(name_list):
            image_id = id

            xml_file = osp.join(ann_dir, name + '.xml')

            with open(xml_file, 'r') as f:
                ann_dict = xmltodict.parse(f.read(), force_list=('object',))

            # Add image item.
            image = self.get_img_item(name + '.jpg', image_id, ann_dict['annotation']['size'])
            images.append(image)

            if 'object' in ann_dict['annotation']:
                for obj in ann_dict['annotation']['object']:
                    # Add annotation item.
                    annotation = self.get_ann_item(obj, image_id, ann_id)
                    annotations.append(annotation)
                    ann_id += 1
            else:
                logger.warning('{} does not have any object'.format(name))

        categories = []
        for name, id in self.cat2id.items():
            # Add category item.
            category = self.get_cat_item(name, id)
            categories.append(category)

        ann = OrderedDict()
        ann['images'] = images
        ann['type'] = 'instances'
        ann['annotations'] = annotations
        ann['categories'] = categories

        logger.info('Saving annotations to {}'.format(save_file))
        with open(save_file, 'w') as f:
            json.dump(ann, f)


if __name__ == '__main__':
    home = os.path.expanduser('~')

    root_datadir = os.path.join(home, 'data/dfsign')
    src_traindir = root_datadir + '/train'
    src_testdir = root_datadir + '/test'
    src_annotation = root_datadir + '/train_label_fix.csv'

    dest_datadir = root_datadir + '/dfsign_chip_voc'
    image_dir = dest_datadir + '/JPEGImages'
    list_dir = dest_datadir + '/ImageSets/Main'
    anno_dir = dest_datadir + '/Annotations'

    coco_dir = root_datadir + '/dfsign_chip_coco'

    converter = PASCALVOC2COCO()
    devkit_path = dest_datadir
    split = 'train'
    save_file = os.path.join(coco_dir, 'annotations/train.json')
    converter.convert(devkit_path, split, save_file)