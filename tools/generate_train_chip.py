# -*- coding: utf-8 -*-
"""
generate 600x600 chips in voc format
image size: 3200x1800
sign size: (0, 500)
"""

import cv2
import random
import os, sys
import glob
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from threading import Lock
from itertools import product
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import utils

import pdb
import traceback

home = os.path.expanduser('~')

root_datadir = os.path.join(home, 'data/dfsign')
src_traindir = root_datadir + '/train'
src_testdir = root_datadir + '/test'
src_annotation = root_datadir + '/train_label_fix.csv'

dest_datadir = root_datadir + '/dfsign_chip_voc'
image_dir = dest_datadir + '/JPEGImages'
list_dir = dest_datadir + '/ImageSets/Main'
anno_dir = dest_datadir + '/Annotations'


def chip_v2(image, gt_boxes, labels):
    """generate chips from a image
    method: random crop around gt_box

    Args:
        image: np.array
        gt_boxes: list of [xmin, ymin, xmax, ymax]
        labels: list of 
    Returns:
        chip list
        new gt_box list
    """
    size = image.shape
    # chip
    chip_list = []
    for box in gt_boxes:
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        # different chip size for different gt size
        if box_w < 50 and box_h < 50:
            chip_size_list = [90, 110, 125, 150, 300]
        elif box_w < 100 and box_h < 100:
            chip_size_list = [180, 250, 350, 500]
        elif box_w < 200 and box_h < 200:
            chip_size_list = [300, 400, 500, 700]
        else:
            chip_size_list = [400, 800, 1000]
        
        for chip_size in chip_size_list:
            # region to random crop around gt
            region_xmin = box[0] - chip_size if box[0] - chip_size > 0 else 0
            region_ymin = box[1] - chip_size if box[1] - chip_size > 0 else 0
            region_xmax = box[0] + chip_size if box[0] + chip_size < 3199 else 3199
            region_ymax = box[1] + chip_size if box[1] + chip_size < 1799 else 1799
            region = np.array([region_xmin, region_ymin, region_xmax, region_ymax])

            # random crop
            while True:
                start_point = 0
                new_x, new_y = region[0], region[1]
                if region[2] - region[0] - chip_size > 0:
                    new_x = region[0] + random.randint(start_point, region[2] - region[0] - chip_size)
                if region[3] - region[1] - chip_size > 0:
                    new_y = region[1] + random.randint(start_point, region[3] - region[1] - chip_size)
                chip = [new_x, new_y, new_x+chip_size, new_y+chip_size]
                # abandon partial overlap chip
                if chip[2] >= box[2] and chip[3] >= box[3]:
                    break
            chip_list.append(np.array(chip))
    
    # chip gt
    chip_gt_list = []
    chip_label_list = []
    for chip in chip_list:
        chip_gt = []
        chip_label = []

        for i, box in enumerate(gt_boxes):
            if utils.overlap(chip, box, 0.8):
                box = [max(box[0], chip[0]), max(box[1], chip[1]), 
                       min(box[2], chip[2]), min(box[3], chip[3])]
                new_box = [box[0] - chip[0], box[1] - chip[1],
                           box[2] - chip[0], box[3] - chip[1]]

                chip_gt.append(np.array(new_box))
                chip_label.append(labels[i])

        chip_gt_list.append(chip_gt)
        chip_label_list.append(chip_label)
    
    return chip_list, chip_gt_list, chip_label_list


def make_xml(chip, box_list, label_list, image_name, tsize):

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_object_num = SubElement(node_root, 'object_num')
    node_object_num.text = str(len(box_list))

    node_location = SubElement(node_root, 'location')
    node_loc_xmin = SubElement(node_location, 'xmin')
    node_loc_xmin.text = str(int(chip[0]) + 1)
    node_loc_ymin = SubElement(node_location, 'ymin')
    node_loc_ymin.text = str(int(chip[1]) + 1)
    node_loc_xmax = SubElement(node_location, 'xmax')
    node_loc_xmax.text = str(int(chip[2]) + 1)
    node_loc_ymax = SubElement(node_location, 'ymax')
    node_loc_ymax.text = str(int(chip[3]) + 1)

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(tsize[0])
    node_height = SubElement(node_size, 'height')
    node_height.text = str(tsize[1])
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i in range(len(box_list)):  
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(label_list[i])
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        # voc dataset is 1-based
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(box_list[i][0]) + 1)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(box_list[i][1]) + 1)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(box_list[i][2] + 1))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(box_list[i][3] + 1))


    xml = tostring(node_root, encoding='utf-8')
    dom = parseString(xml)
    # print(xml)
    return dom

def write_chip_and_anno(image, imgid, 
    chip_list, chip_gt_list, chip_label_list):
    """write chips of one image to disk and make xml annotations
    """
    assert len(chip_gt_list) > 0
    for i, chip in enumerate(chip_list):
        img_name = '%s_%d.jpg' % (imgid, i)
        xml_name = '%s_%d.xml' % (imgid, i)

        # target size
        tsize = (600, 600)
        # resize ratio -> target size
        ratio_w = (chip[2] - chip[0]) / tsize[0]
        ratio_h = (chip[3] - chip[1]) / tsize[1]
        
        chip_img = image[chip[1]:chip[3], chip[0]:chip[2], :].copy()
        chip_img = cv2.resize(chip_img, tsize, interpolation=cv2.INTER_LINEAR)

        bbox = []
        for gt in chip_gt_list[i]:
            bbox.append([gt[0] / ratio_w,
                        gt[1] / ratio_h,
                        gt[2] / ratio_w,
                        gt[3] / ratio_h])
        bbox = np.array(bbox, dtype=np.int)

        dom = make_xml(chip, bbox, chip_label_list[i], img_name, tsize)

        cv2.imwrite(os.path.join(image_dir, img_name), chip_img)
        with open(os.path.join(anno_dir, xml_name), 'w') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8'))


def get_box_label(label_df, im_name):
    boxes = []
    labels = []
    im_label = label_df[label_df.filename == im_name]
    for index, row in im_label.iterrows():
        xmin = min(row['X1'], row['X2'], row['X3'], row['X4'])
        ymin = min(row['Y1'], row['Y2'], row['Y3'], row['Y4'])
        xmax = max(row['X1'], row['X2'], row['X3'], row['X4'])
        ymax = max(row['Y1'], row['Y2'], row['Y3'], row['Y4'])

        xmax = min(xmax, 3200-1)
        ymax = min(ymax, 1800-1)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(row['type'])
    return np.array(boxes), labels


def generate_imgset(train_list):
    # train_list = os.listdir(image_dir)
    train_list = [x.split('.')[0] for x in train_list]
    with open(os.path.join(list_dir, 'train.txt'), 'w') as f:
        f.writelines([x + '\n' for x in train_list])

def _worker(imgid, label_df):
    try:
        image = cv2.imread(os.path.join(src_traindir, imgid+'.jpg'))
        gt_boxes, labels = get_box_label(label_df, imgid+'.jpg')

        chip_list, chip_gt_list, chip_label_list = chip_v2(image, gt_boxes, labels)
        write_chip_and_anno(image, imgid, chip_list, chip_gt_list, chip_label_list)
        return len(chip_list)
        # _progress()
    except Exception:
        traceback.print_exc()
        os._exit(0) 

def main():
    if not os.path.exists(dest_datadir):
        os.mkdir(dest_datadir)
        os.mkdir(image_dir)
        os.makedirs(list_dir)
        os.mkdir(anno_dir)
    
    train_list = glob.glob(src_traindir + '/*.jpg')
    random.shuffle(train_list)
    # train_list, val_list = train_list[:-2000], train_list[-2000:]
    train_list = [os.path.basename(x)[:-4] for x in train_list]

    # read label
    df = pd.read_csv(src_annotation)

    train_chip_ids = []
    for i, img_id in enumerate(train_list):
        sys.stdout.write('\rsearch: {:d}/{:d} {:s}'
                        .format(i + 1, len(train_list), train_list[i]))
        sys.stdout.flush()

        chiplen = _worker(img_id, df)
        for i in range(chiplen):
            train_chip_ids.append('%s_%s' % (img_id, i))
    
    generate_imgset(train_chip_ids)

if __name__ == '__main__':
    main()
