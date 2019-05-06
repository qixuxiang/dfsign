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
    boxes = []
    im_label = label_df[label_df.filename == im_name]
    for index, row in im_label.iterrows():
        xmin = min(row['X1'], row['X2'], row['X3'], row['X4'])
        ymin = min(row['Y1'], row['Y2'], row['Y3'], row['Y4'])
        xmax = max(row['X1'], row['X2'], row['X3'], row['X4'])
        ymax = max(row['Y1'], row['Y2'], row['Y3'], row['Y4'])
        boxes.append([xmin, ymin, xmax, ymax])
    return np.array(boxes)


def _generate_mask(img_path, label_df):
    try:
        # image mask
        img_name = os.path.split(img_path)[-1]

        width, height = 3200, 1800
        # chip mask 40x23, model input size 640x320
        mask_w, mask_h = 50, 29

        region_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        boxes = get_box(label_df, img_name)
        for box in boxes:
            xmin = np.floor(1.0 * box[0] / width * mask_w).astype(np.int32)
            ymin = np.floor(1.0 * box[1] / height * mask_h).astype(np.int32)
            xmax = np.floor(1.0 * box[2] / width * mask_w).astype(np.int32)
            ymax = np.floor(1.0 * box[3] / height * mask_h).astype(np.int32)
            region_mask[ymin : ymax+1, xmin : xmax+1] = 1
        maskname = os.path.join(segmentation_dir, img_name[:-4] + '_region.png')
        cv2.imwrite(maskname, region_mask)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    if not os.path.exists(dest_datadir):
        os.mkdir(dest_datadir)
        os.mkdir(image_dir)
        os.mkdir(segmentation_dir)
        os.mkdir(list_folder)

    train_list = glob.glob(src_traindir + '/*.jpg')
    test_list = glob.glob(src_testdir + '/*.jpg')
    random.shuffle(train_list)
    train_list, val_list = train_list[:-1000], train_list[-1000:]
    all_list = train_list + val_list

    # write list file
    with open(os.path.join(list_folder, 'train.txt'), 'w') as f:
        temp = [os.path.basename(x)[:-4]+'\n' for x in train_list]
        f.writelines(temp)
    with open(os.path.join(list_folder, 'val.txt'), 'w') as f:
        temp = [os.path.basename(x)[:-4]+'\n' for x in val_list]
        f.writelines(temp)
    # with open(os.path.join(list_folder, 'test.txt'), 'w') as f:
    #     temp = [os.path.basename(x)[:-4]+'\n' for x in test_list]
    #     f.writelines(temp)

    # print('copy image....')
    # with concurrent.futures.ThreadPoolExecutor() as exector:
    #     exector.map(_copy, train_list, [image_dir]*len(train_list))
    # print('done.')

    # read label
    df = pd.read_csv(src_annotation)

    print('generate mask...')
    with concurrent.futures.ThreadPoolExecutor() as exector:
        exector.map(_generate_mask, all_list, [df]*len(all_list))
    print('done.')
