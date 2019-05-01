# -*- coding: utf-8 -*-
"""
generate 300x300 chips in voc format
image size: 2048x2048
sign size: (0, 500)
"""

import cv2
import json
import os, sys
import numpy as np
import concurrent.futures
from tqdm import tqdm
from threading import Lock
from random import randint
from itertools import product
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import utils

import pdb
import traceback

home = os.path.expanduser('~')
root_datadir = os.path.join(home, 'data/TT100K')
src_traindir = root_datadir + '/data/train'
src_testdir = root_datadir + '/data/test'
src_annotation = root_datadir + '/data/annotations.json'

# TT100K imageset list files
train_ids = src_traindir + '/ids.txt'
test_ids = src_testdir + '/ids.txt'

dest_datadir = root_datadir + '/TT100K_chip_voc'
image_dir = dest_datadir + '/JPEGImages'
list_dir = dest_datadir + '/ImageSets/Main'
anno_dir = dest_datadir + '/Annotations'

# add path
sys.path.append(os.path.join(root_datadir, 'code/python'))
import anno_func

if not os.path.exists(dest_datadir):
    os.mkdir(dest_datadir)
    os.mkdir(image_dir)
    os.makedirs(list_dir)
    os.mkdir(anno_dir)


def chip_v1(image, gt_boxes):
    """generate chips from a image
    method: random crop in whole image

    Args:
        image: np.array
        gt_boxes: list of [xmin, ymin, xmax, ymax]
    
    Returns:
        chip list, size 300x300 
        new gt_box list
    """
    # use 600x600 chip
    use_600 = False
    w = [box[2] - box[0] for box in gt_boxes]
    h = [box[3] - box[1] for box in gt_boxes]
    if np.max(w) > 200 or np.max(h) > 200:
        use_600 = True

    size = image.shape
    for i, j in product(range(size[1]-300, 30), 
                        range(size[0]-300, 30)):
        pass


def chip_v2(image, gt_boxes, labels):
    """generate chips from a image
    method: random crop around gt_box

    Args:
        image: np.array
        gt_boxes: list of [xmin, ymin, xmax, ymax]
        labels: list of 
    Returns:
        chip list, size 300x300 
        new gt_box list
    """
    size = image.shape
    # chip
    chip_list = []
    for box in gt_boxes:
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        # different chip size for different gt size
        if box_w < 100 and box_h < 100:
            chip_size_list = [150, 300]
        elif box_w < 300 and box_h < 300:
            chip_size_list = [300, 600]
        else:
            chip_size_list = [600, 800]
        
        for chip_size in chip_size_list:
            # region to random crop around gt
            region = np.clip( 
                [box[0] - chip_size, box[1] - chip_size,
                box[0] + chip_size, box[1] + chip_size],
                0, 2047)

            # random crop
            while True:
                start_point = 0
                new_x, new_y = region[0], region[1]
                if region[2] - region[0] - chip_size > 0:
                    new_x = region[0] + randint(start_point, region[2] - region[0] - chip_size)
                if region[3] - region[1] - chip_size > 0:
                    new_y = region[1] + randint(start_point, region[3] - region[1] - chip_size)
                chip = [new_x, new_y, new_x+chip_size, new_y+chip_size]
                # abandon partial overlap chip
                if chip[2] >= box[2] and chip[3] >= box[3]:
                    break
                start_point += 10
            chip_list.append(np.array(chip))
    
    # chip gt
    chip_gt_list = []
    chip_label_list = []
    for chip in chip_list:
        chip_gt = []
        chip_label = []

        for i, box in enumerate(gt_boxes):
            if utils.overlap(chip, box, 0.6):
                box = [max(box[0], chip[0]), max(box[1], chip[1]), 
                       min(box[2], chip[2]), min(box[3], chip[3])]
                new_box = [box[0] - chip[0], box[1] - chip[1],
                           box[2] - chip[0], box[3] - chip[1]]

                chip_gt.append(np.array(new_box))
                chip_label.append(labels[i])

        chip_gt_list.append(chip_gt)
        chip_label_list.append(chip_label)
    
    return chip_list, chip_gt_list, chip_label_list


def make_xml(chip, box_list, label_list, image_name):

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
    node_width.text = '416'
    node_height = SubElement(node_size, 'height')
    node_height.text = '416'
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
    for i, chip in enumerate(chip_list):
        img_name = '%d_%d.jpg' % (imgid, i)
        xml_name = '%d_%d.xml' % (imgid, i)

        # resize ratio -> 300x300
        ratio = (chip[2] - chip[0]) / 416
        
        chip_img = image[chip[1]:chip[3], chip[0]:chip[2], :].copy()
        chip_img = cv2.resize(chip_img, (416, 416), interpolation=cv2.INTER_LINEAR)

        dom = make_xml(chip, chip_gt_list[i] / ratio, chip_label_list[i], img_name)

        cv2.imwrite(os.path.join(image_dir, img_name), chip_img)
        with open(os.path.join(anno_dir, xml_name), 'w') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8'))


def get_box_label(annos, imgid):
    valid_label = utils.TT100K_CLASSES
    img = annos["imgs"][imgid]
    box_all = []
    label_all = []
    for obj in img['objects']:
        box = obj['bbox']
        if obj['category'] in valid_label:
            box = [int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])]
            box_all.append(np.clip(box, 0, 2047))
            label_all.append(obj['category'])
    return box_all, label_all


def generate_imgset(train_list):
    # train_list = os.listdir(image_dir)
    train_list = [x.split('.')[0] for x in train_list]
    with open(os.path.join(list_dir, 'train.txt'), 'w') as f:
        f.writelines([x + '\n' for x in train_list])

numTag = 0
lock = Lock()
def _progress():
    global numTag
    with lock:
        numTag += 1
        sys.stdout.write('\r{0}'.format(str(numTag)))
        sys.stdout.flush()

def _worker(imgid, annos):
    try:
        image = cv2.imread(os.path.join(src_traindir, imgid+'.jpg'))
        gt_boxes, labels = get_box_label(annos, imgid)
        chip_list, chip_gt_list, chip_label_list = chip_v2(image, gt_boxes, labels)
        write_chip_and_anno(image, int(imgid), chip_list, chip_gt_list, chip_label_list)
        return len(chip_list)
        # _progress()
    except Exception:
        traceback.print_exc()
        print(imgid)
        os._exit(0) 

def main():
    with open(src_annotation, 'r') as f:
        annos = json.load(f)
    
    with open(train_ids, 'r') as f:
        train_list = [x.strip() for x in f.readlines()]

    # _worker('12397', annos)

    # with concurrent.futures.ThreadPoolExecutor() as exector:
    #     exector.map(_worker, train_list, [annos] * len(train_list))
    train_chip_ids = []
    for img_id in tqdm(train_list):
        chiplen = _worker(img_id, annos)
        for i in range(chiplen):
            train_chip_ids.append('%s_%s' % (img_id, i))

    generate_imgset(train_chip_ids)
    # with open(os.path.join(list_dir, 'train.txt'), 'r') as f:
    #     chip = [x.split('_')[0] for x in f.readlines()]
    # with open(train_ids, 'r') as f:
    #     img = [x.strip() for x in f.readlines()]
    # for x in img:
    #     if x not in chip:
    #         print(x)

if __name__ == '__main__':
    main()
