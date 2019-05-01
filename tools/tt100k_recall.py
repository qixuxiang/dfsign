"""
recall of segmentation result of tt100k
"""

import os, sys
import numpy as np
import cv2 as cv
from glob import glob
import json
from tqdm import tqdm
from operator import mul
import utils
import pdb


user_home = os.path.expanduser('~')
datadir = os.path.join(user_home, 'data/TT100K')
label_path = datadir + '/TT100K_voc/SegmentationClass'
annos_path = datadir + '/data/annotations.json'
image_path = datadir + '/TT100K_voc/JPEGImages'
mask_path = os.path.join(user_home, 'codes/gluon-cv/projects/seg/outdir')
            # 'codes/deeplab-tensorflow/deeplab/datasets/tt100k/exp/vis/raw_segmentation_results')


def get_box(annos, imgid):
    img = annos["imgs"][imgid]
    box_all = []
    for obj in img['objects']:
        box = obj['bbox']
        box = [int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])]
        # box = [int(x * 0.3) for x in box]
        box_all.append(box)
    return box_all


def _boxvis(mask, mask_box):
    ret, binary = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
    print(mask_box)
    for box in mask_box:
        cv.rectangle(binary, (box[0], box[1]), (box[2], box[3]), 100, 2)
    cv.imshow('a', binary)
    key = cv.waitKey(0)
    sys.exit(0)


def vis_undetected_image(img_list):
    annos = json.loads(open(annos_path).read())

    for image in img_list:
        mask_file = os.path.join(mask_path, image+'.png')
        image_file = os.path.join(image_path, image+'.jpg')

        mask_img = cv.imread(mask_file, cv.IMREAD_GRAYSCALE)
        original_img = cv.imread(image_file)
        original_img[:,:,1] = np.clip(original_img[:,:,1] + mask_img*70, 0, 255)

        label_box = get_box(annos, image)
        for box in label_box:
            cv.rectangle(original_img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 1, 1)
        
        cv.imshow('1', original_img)
        key = cv.waitKey(1000*100)
        if key == 27:
            break

def main():
    annos = json.loads(open(annos_path).read())

    label_object = []
    detect_object = []
    mask_object = []
    undetected_img = []
    pixel_num = []
    for raw_file in tqdm(glob(mask_path + '/*.png')):
        img_name = os.path.basename(raw_file)
        imgid = os.path.splitext(img_name)[0]
        label_file = os.path.join(label_path, img_name)
        image_file = os.path.join(image_path, imgid + '.jpg')
        
        mask_img = cv.imread(raw_file, cv.IMREAD_GRAYSCALE)
        # mask_img = cv.resize(mask_img, (2048, 2048), interpolation=cv.INTER_LINEAR)
        pixel_num.append(np.sum(mask_img))

        height, width = mask_img.shape[:2]

        label_box = get_box(annos, imgid)
        mask_box = utils.generate_box_from_mask(mask_img)
        mask_box = list(map(utils.resize_box, mask_box, 
                        [width]*len(mask_box), [2048]*len(mask_box)))
        mask_box = utils.enlarge_box(mask_box, (2048, 2048), ratio=2)
        # _boxvis(mask_img, mask_box)
        # break

        count = 0
        for box1 in label_box:
            for box2 in mask_box:
                if utils.overlap(box2, box1):
                    count += 1
                    break

        label_object.append(len(label_box))
        detect_object.append(count)
        mask_object.append(len(mask_box))
        if len(label_box) != count:
            undetected_img.append(imgid)

    print('recall: %f' % (np.sum(detect_object) / np.sum(label_object)))
    print('cost avg: %f, std: %f' % (np.mean(pixel_num), np.std(pixel_num)))
    print('detect box avg: %f, std %d' %(np.mean(mask_object), np.std(mask_object)))
    # print(undetected_img)

if __name__ == '__main__':
    img_list = ['1883', '75227', '38108', '29501', '48010', '15366', '91027', '31998', '25647', '29849', '77147', '82748', '78591', '25503', '75494', '36342', '13802', '1774', '66010', '83208', '51875', '27645', '74380', '12186', '37304', '25333', '50648', '40008', '28988', '44609', '18935', '9662', '36686', '68353', '75883', '41462', '2706', '26042', '35419', '14724', '6599', '29618', '51142', '80064', '75881', '77156', '65419', '37771', '66589', '4902', '1792', '16489', '81686', '89739', '70814', '58194', '42891', '88783', '12467', '40363', '76771', '36393', '5998', '90375', '97043', '90326', '2736', '22779', '36302', '79023', '79595', '77713', '72001', '84082', '84626', '94686', '38711', '69700', '66004', '94893', '94910', '94297', '7807', '70386', '94598', '71249', '70437', '20942', '41372', '40935', '92518', '4232', '39610', '97625', '97290', '96726', '58281', '80925', '62728', '38273', '41321', '67372', '68066', '5773', '25421', '76568', '56576', '58719', '45154', '94307', '31641', '47787', '12339', '73250', '46346', '62025', '59076', '2', '51435', '38199', '13931', '8783', '29297', '86217', '29714', '20600', '5051', '83167', '27986', '23320', '39338', '86066', '52335', '12883', '27663', '96461', '8099', '96069', '34782', '64497', '74315', '64195', '63555', '15332', '74518', '27009', '62713', '53500', '92707', '26234', '74668', '31643', '66117']
    # vis_undetected_image(img_list)
    main()
