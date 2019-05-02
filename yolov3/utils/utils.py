import glob
import random
from collections import defaultdict

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils import torch_utils
from utils.focal_loss import FocalLoss

import pdb

matplotlib.rc('font', **{'family': 'normal', 'size': 11})

# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def float3(x):  # format floats to 3 decimals
    return float(format(x, '.3f'))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def load_classes(path):
    # Loads class labels at 'path'
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def model_info(model):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %40s %9s %12g %20s %10.3g %10.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (i + 1, n_p, n_g))


def coco_class_weights():  # frequency of each class in coco train2014
    weights = 1 / torch.FloatTensor(
        [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
         6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
         4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
         5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
         1877, 17630, 4337, 4624, 1075, 3468, 135, 1380])
    weights /= weights.sum()
    return weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_coords(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain = float(img_size) / max(img0_shape)  # gain  = old / new
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = np.clip(coords[:, :4], a_min=0, a_max=None)
    return coords


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

            # Plot
            # plt.plot(recall_curve, precision_curve)

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou


def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


def compute_loss(p, targets):  # predictions, targets
    FT = torch.cuda.FloatTensor if p[0].is_cuda else torch.FloatTensor
    lxy, lwh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0])
    txy, twh, tcls, indices, ignores = targets

    MSE = nn.MSELoss()
    CE = nn.CrossEntropyLoss()
    SML1 = nn.SmoothL1Loss()
    xBCE = nn.BCEWithLogitsLoss()
    FL = FocalLoss(gamma=2, alpha=0.85)

    # Compute losses
    # gp = [x.numel() for x in tconf]  # grid points
    for i, pi0 in enumerate(p):  # layer i predictions, i
        b, a, gj, gi = indices[i]  # image, anchor, gridx, gridy
        ignore_b, ignore_anchor, ignore_gj, ignore_gi = ignores[i]
        
        tconf = torch.zeros_like(pi0[..., 0])  # conf
        ignore_mask = torch.ones_like(pi0[..., 0])  #ignore

        # Compute losses
        k = 1  # nT / bs
        if len(b) > 0:
            pi = pi0[b, a, gj, gi]  # predictions closest to anchors
            tconf[b, a, gj, gi] = 1  # conf
            ignore_mask[ignore_b, ignore_anchor, ignore_gj, ignore_gi] = 0

            lxy += (k * 8) * MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy loss  
            lwh += (k * 4) * MSE(pi[..., 2:4], twh[i])  # wh yolo loss
            lcls += (k * 1) * CE(pi[..., 5:], tcls[i])  # class_conf loss

        # BCE = nn.BCEWithLogitsLoss(weight=ignore_mask)
        # lconf += (k * 64) * BCE(pi0[..., 4], tconf)
        lconf += (k * 512) * FL(torch.sigmoid(pi0[..., 4]), tconf, ignore_mask)
    loss = lxy + lwh + lconf + lcls

    # Add to dictionary
    d = defaultdict(float)
    losses = [loss.item(), lxy.item(), lwh.item(), lconf.item(), lcls.item()]
    for k, v in zip(['total', 'xy', 'wh', 'conf', 'cls'], losses):
        d[k] = v

    return loss, d


def build_targets(model, targets):
    # targets = [image, class, x, y, w, h]
    if isinstance(model, nn.parallel.DataParallel):
        model = model.module

    img_index, cls = targets[:, :2].long().t() # target image, class

    txy, twh, tcls, indices, ignores = [], [], [], [], []
    for i in model.yolo_layers:
        layer = model.module_list[i][0]
        _, _, anchor_vec, _ = layer.create_grids(layer.nG.device)

        # scale to grid size
        label_xy = targets[:, 2:4] * layer.nG
        label_wh = targets[:, 4:6] * layer.nG

        # iou of targets-anchors
        iou = torch.stack([wh_iou(x, label_wh) for x in anchor_vec], 0)
        max_iou, max_anchor = iou.max(0)  # best iou and anchor

        # reject below threshold ious (OPTIONAL)
        reject_thresh = 0.10
        j = max_iou > reject_thresh
        gxy = label_xy[j]
        gwh = label_wh[j]
        b, c, a = img_index[j], cls[j], max_anchor[j]

        # Indices
        gi, gj = gxy.long().t()  # grid_i, grid_j
        indices.append((b, a, gj, gi))

        # ignore iou > 0.5 but not max
        mask = ((iou > 0.5) & (iou < max_iou)).nonzero()
        ignore_anchor, j = mask[:, 0], mask[:, 1]
        ignore_gxy = label_xy[j]
        ignore_b = img_index[j]
        gi, gj = ignore_gxy.long().t()
        ignores.append((ignore_b, ignore_anchor, gj, gi))
        
        # XY coordinates
        txy.append(gxy - gxy.floor())

        # Width and height
        twh.append(torch.log(gwh / anchor_vec[a]))  # yolo method
        # twh.append((gwh / layer.anchor_vec[a]) ** (1 / 3) / 2)  # power method

        # Class
        tcls.append(c)
        assert c.max() <= layer.nC, 'Target classes exceed model classes'

    return txy, twh, tcls, indices, ignores


def iou_calc1(boxes1, boxes2):
    """
    array
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + 1e-16 + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def nms(prediction, score_threshold=0.5, iou_threshold=0.4, sigma=0.3, method='nms'):
    """
    :param prediction:
    (x, y, w, h, conf, c*N)
    :return: best_bboxes
    假设NMS后剩下N个bbox，那么best_bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入训练尺寸的，score = conf * prob，class是bbox所属类别的索引号
    """
    prediction = prediction.cpu().numpy()
    output = [np.array([]) for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # ignore conf < thresh
        pred = pred[pred[:, 4] > 0.001]

        class_pred = pred[:, 5:].argmax(axis=1)
        class_prob = pred[:, 5:].max(axis=1)

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        # Detections ordered as (x1, y1, x2, y2, obj_conf * class_prob, class_pred)
        detections = np.concatenate((pred[:, :4], (pred[:, 4]*class_prob)[:, np.newaxis], class_pred[:, np.newaxis]), 1)
        # sort by score 
        detections = detections[(-detections[:,4]).argsort()[:1000]]
        # Iterate through all predicted classes
        unique_labels = np.unique(detections[:, -1])

        best_bboxes = []
        for cls in unique_labels:
            cls_mask = (detections[:, 5] == cls)
            cls_bboxes = detections[cls_mask]
            while len(cls_bboxes) > 0:
                best_bbox = cls_bboxes[0]
                best_bboxes.append(best_bbox)
                cls_bboxes = cls_bboxes[1:]
                iou = iou_calc1(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                assert method in ['nms', 'soft-nms']
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight = np.ones((len(iou),), dtype=np.float32)
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > score_threshold
                cls_bboxes = cls_bboxes[score_mask]
        if len(best_bboxes) > 0:
            best_bboxes = np.array(best_bboxes)
            # Add max detections to outputs
            output[image_i] = best_bboxes

    return output


def strip_optimizer_from_checkpoint(filename='weights/best.pt'):
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    a = torch.load(filename, map_location='cpu')
    a['optimizer'] = []
    torch.save(a, filename.replace('.pt', '_lite.pt'))


def coco_class_count(path='../coco/labels/train2014/'):
    # Histogram of occurrences per class
    nC = 80  # number classes
    x = np.zeros(nC, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nC)
        print(i, len(files))


def coco_only_people(path='../coco/labels/val2014/'):
    # Find images with only people
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def plot_wh_methods():  # from utils.utils import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, ya, '.-', label='yolo method')
    plt.plot(x, yb ** 2, '.-', label='^2 power method')
    plt.plot(x, yb ** 2.5, '.-', label='^2.5 power method')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.jpg', dpi=fig.dpi)


def plot_results(start=0, stop=0):  # from utils.utils import *; plot_results()
    # Plot training results files 'results*.txt'
    # import os; os.system('wget https://storage.googleapis.com/ultralytics/yolov3/results_v3.txt')

    fig = plt.figure(figsize=(14, 7))
    s = ['X + Y', 'Width + Height', 'Confidence', 'Classification', 'Train Loss', 'Precision', 'Recall', 'mAP', 'F1',
         'Test Loss']
    for f in sorted(glob.glob('results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 9, 10, 11, 12, 13]).T
        x = range(start, stop if stop else results.shape[1])
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.plot(x, results[i, x].clip(max=500), marker='.', label=f)
            plt.title(s[i])
            if i == 0:
                plt.legend()
    fig.tight_layout()
    fig.savefig('results.jpg', dpi=fig.dpi)
