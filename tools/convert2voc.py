"""convert VOC format
+ VOC2012
    + JPEGImages
    + SegmentationClass
"""

import os, sys
import glob
import cv2
import random
import shutil
import numpy as np
import pandas as pd
import concurrent.futures
import pdb

random.seed(100)
userhome = os.path.expanduser('~')

src_datadir = os.path.join(userhome, 'data/dfsign')
src_traindir = src_datadir + '/train'
src_testdir = src_datadir + '/test'
src_annotation = src_datadir + '/train_label_fix.csv'

dest_datadir = os.path.join(userhome, 'data/dfsign/dfsign_region_voc')
image_dir = dest_datadir + '/JPEGImages'
segmentation_dir = dest_datadir + '/SegmentationClass'
list_folder = dest_datadir + '/ImageSets'

# copy train and test images
def _copy(src_image, dest_path):
    shutil.copy(src_image, dest_path)

def _resize(src_image, dest_path):
    img = cv2.imread(src_image)

    height, width = img.shape[:2]
    size = (int(width), int(height))

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    name = os.path.basename(src_image)
    cv2.imwrite(os.path.join(dest_path, name), img)

def get_box(label_df, im_name):
    im_label = label_df[label_df.filename == im_name]
    for index, row in im_label.iterrows():
        print(type(row['X1']))
        xmin = min(row['X1'], row['X2'], row['X3'], row['X4'])
        xmax = max(row['X1'], row['X2'], row['X3'], row['X4'])

# mask
def _generate_mask(img_path):
    try:
        # image mask
        img_id = os.path.split(img_path)[-1][:-4]
        im_data = anno_func.load_img(annos, src_datadir, img_id)
        mask = anno_func.load_mask(annos, src_datadir, img_id, im_data)

        height, width = mask.shape[:2]
        size = (int(width), int(height))

        mask = cv2.resize(mask, size)
        maskname = os.path.join(segmentation_dir, img_id + '.png')
        cv2.imwrite(maskname, mask)

        # chip mask 30x30
        chip_mask = np.zeros((30, 30), dtype=int)
        boxes = get_box(annos, img_id)
        for box in boxes:
            xmin, ymin, xmax, ymax = np.floor(box * 30).astype(np.int32)
            chip_mask[ymin : ymax+1, xmin : xmax+1] = 1
        maskname = os.path.join(segmentation_dir, img_id + '_chip.png')
        cv2.imwrite(maskname, chip_mask)

    except Exception as e:
        print(e)

# print('mask...')
# with concurrent.futures.ThreadPoolExecutor() as exector:
#     exector.map(_generate_mask, all_list)
# _generate_mask(all_list[0])

if __name__ == "__main__":
    if not os.path.exists(dest_datadir):
        os.mkdir(dest_datadir)
        os.mkdir(image_dir)
        os.mkdir(segmentation_dir)
        os.mkdir(list_folder)

    train_list = glob.glob(src_traindir + '/*.jpg')
    random.shuffle(train_list)
    train_list, val_list = train_list[:-2000], train_list[-2000:]
    all_list = train_list + val_list

    # print('copy image....\n')
    # with concurrent.futures.ThreadPoolExecutor() as exector:
    #     exector.map(_copy, all_list, [image_dir]*len(all_list))

    # read label
    df = pd.read_csv(src_annotation)
    get_box(df, '000015983ee24b9bb06f0a493e40d396.jpg')