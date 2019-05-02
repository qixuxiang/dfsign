import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import xyxy2xywh
from .transforms import *


class LoadImages():  # for inference
    def __init__(self, path, img_size=416):
        self.height = img_size
        img_formats = ['.jpg', '.jpeg', '.png', '.tif']
        vid_formats = ['.mov', '.avi', '.mp4']

        files = []
        if isinstance(path, list):
            files = path
        elif os.path.isdir(path):
            files = sorted(glob.glob('%s/*.*' % path))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return len(self.path)

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'File Not Found ' + path
            # print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img, _, = letterbox(img0, None, height=self.height, mode='test')

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = normalize(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, img_size=416):
        self.cam = cv2.VideoCapture(0)
        self.height = img_size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == 27:  # esc to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Read image
        ret_val, img0 = self.cam.read()
        assert ret_val, 'Webcam Error'
        img_path = 'webcam_%g.jpg' % self.count
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, mode='test')

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = normalize(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadImagesAndLabels(Dataset):  # for training
    def __init__(self, img_files, label_files=None, img_size=608, mode='train'):
        self.img_files = img_files
        self.label_files = label_files if label_files else \
                            [x.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]
        self.nF = len(self.img_files)  # number of image files
        self.height = img_size
        self.mode = mode

        assert self.nF > 0, 'No images found in %s' % path

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # read img and label
        img = cv2.imread(self.img_files[index])  # BGR
        assert img is not None, 'File Not Found ' + self.img_files[index]
        h, w = img.shape[:2]

        labels = self._load_label(self.label_files[index])
        if self.mode == 'train':
            # hsv
            img = augment_hsv(img, fraction=0.5)
            # random crop
            labels, crop = random_crop_with_constraints(labels, (w, h))
            img = img[crop[1]:crop[1]+crop[3], crop[0]:crop[0]+crop[2], :].copy()
            # pad and resize
            img, labels = letterbox(img, labels, height=self.height, mode=self.mode)
            # Augment image and labels
            img, labels = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
            # random left-right flip
            img, labels = random_flip(img, labels, 0.5)
            # color distort
            # img = random_color_distort(img)
        else:
            # pad and resize
            img, labels = letterbox(img, labels, height=self.height, mode=self.mode)

        # show_image(img, labels)

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels = np.clip(labels, 0, self.height - 1)
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy()) / self.height

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = normalize(img)

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = torch.from_numpy(img).float()
        labels_out = labels_out.float()
        shape = np.array([h,w], dtype=np.float32)
        return (img, labels_out, shape, self.img_files[index])
    
    @staticmethod
    def collate_fn(batch):
        img, label, hw, path = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), hw, path

    def _load_label(self, label_path):
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 1] = labels0[:, 1] - labels0[:, 3] / 2
            labels[:, 2] = labels0[:, 2] - labels0[:, 4] / 2
            labels[:, 3] = labels0[:, 1] + labels0[:, 3] / 2
            labels[:, 4] = labels0[:, 2] + labels0[:, 4] / 2
        else:
            labels = np.array([])
        return labels

    def __len__(self):
        return self.nF  # number of batches


def convert_tif2bmp(p='../xview/val_images_bmp'):
    import glob
    import cv2
    files = sorted(glob.glob('%s/*.tif' % p))
    for i, f in enumerate(files):
        print('%g/%g' % (i + 1, len(files)))
        cv2.imwrite(f.replace('.tif', '.bmp'), cv2.imread(f))
        os.system('rm -rf ' + f)

def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '-')
    plt.show()
