"""
check train chips labels
"""

import os
import sys
import cv2
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import xml.etree.ElementTree as ET

import pdb

home = os.path.expanduser('~')
root_datadir = os.path.join(home, 'data/dfsign')
src_traindir = root_datadir + '/train'
src_testdir = root_datadir + '/test'
src_annotation = root_datadir + '/train_label_fix.csv'

dest_datadir = root_datadir + '/dfsign_chip_voc'
image_dir = dest_datadir + '/JPEGImages'
list_dir = dest_datadir + '/ImageSets/Main'
anno_dir = dest_datadir + '/Annotations'

def parse_xml(file):
    xml = ET.parse(file).getroot()
    box_all = []
    pts = ['xmin', 'ymin', 'xmax', 'ymax']

    # bounding boxes
    for obj in xml.iter('object'):
        bbox = obj.find('bndbox')
        
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            bndbox.append(cur_pt)
        box_all += [bndbox]
    return box_all

def get_box_label(label_df, im_name):
    boxes = []
    labels = []
    im_label = label_df[label_df.filename == im_name]
    for index, row in im_label.iterrows():
        xmin = min(row['X1'], row['X2'], row['X3'], row['X4'])
        ymin = min(row['Y1'], row['Y2'], row['Y3'], row['Y4'])
        xmax = max(row['X1'], row['X2'], row['X3'], row['X4'])
        ymax = max(row['Y1'], row['Y2'], row['Y3'], row['Y4'])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(row['type'])
    return np.array(boxes), labels


def _boxvis(img, gt_box_list):
    img1 = img.copy()
    for box in gt_box_list:
        cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)

    plt.subplot(1, 1, 1); plt.imshow(img1[:, :, [2,1,0]])
    plt.show()
    cv2.waitKey()
    

def main():
    # df = pd.read_csv(src_annotation)

    # for name in df.filename:
    #     img = cv2.imread(os.path.join(src_traindir, name))
    #     box, _ = get_box_label(df, name)
    #     _boxvis(img, box)

    with open(os.path.join(list_dir, 'train.txt'), 'r') as f:
        img_list = [x.strip() for x in f.readlines()]

    for name in img_list:
        img = cv2.imread(os.path.join(image_dir, name+'.jpg'))
        box = parse_xml(os.path.join(anno_dir, name+'.xml'))
        _boxvis(img, box)

if __name__ == '__main__':
    main()
