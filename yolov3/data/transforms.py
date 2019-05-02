"""
data aug methods
"""
import os
import math
import cv2
import random
import numpy as np
from utils.utils import iou_calc1


def augment_hsv(img, fraction):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = img_hsv[:, :, 1].astype(np.float32)
    V = img_hsv[:, :, 2].astype(np.float32)

    a = (random.random() * 2 - 1) * fraction + 1
    S *= a
    if a > 1:
        np.clip(S, a_min=0, a_max=255, out=S)

    a = (random.random() * 2 - 1) * fraction + 1
    V *= a
    if a > 1:
        np.clip(V, a_min=0, a_max=255, out=V)

    img_hsv[:, :, 1] = S.astype(np.uint8)
    img_hsv[:, :, 2] = V.astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
    return img


def letterbox(img, labels, height=416, mode='train', color=(127.5, 127.5, 127.5)):
    """
    resize a rectangular image to a padded square
    """
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    if mode == 'test':
        dw = (max(shape) - shape[1]) / 2  # width padding
        dh = (max(shape) - shape[0]) / 2  # height padding
        left, right = round(dw - 0.1), round(dw + 0.1)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
    else:
        dw = random.randint(0, max(shape) - shape[1])
        dh = random.randint(0, max(shape) - shape[0])
        left, right = dw, max(shape) - shape[1] - dw
        top, bottom = dh, max(shape) - shape[0] - dh
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    interp = np.random.randint(0, 5)
    img = cv2.resize(img, (height, height), interpolation=interp)  # resized, no border
    if labels is not None and len(labels) > 0:
        labels[:, 1] = ratio * (labels[:, 1] + left)
        labels[:, 2] = ratio * (labels[:, 2] + top)
        labels[:, 3] = ratio * (labels[:, 3] + left)
        labels[:, 4] = ratio * (labels[:, 4] + top)

    return img, labels


def random_affine(img, targets=(), degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if len(targets) > 0:
        n = targets.shape[0]
        points = targets[:, 1:5].copy()
        area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # apply angle-based reduction of bounding boxes
        radians = a * math.pi / 180
        reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        x = (xy[:, 2] + xy[:, 0]) / 2
        y = (xy[:, 3] + xy[:, 1]) / 2
        w = (xy[:, 2] - xy[:, 0]) * reduction
        h = (xy[:, 3] - xy[:, 1]) * reduction
        xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        np.clip(xy, 0, height, out=xy)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return imw, targets

def random_flip(img, labels, px=0.5):
    """
    random horizontal flip
    """
    height = img.shape[0]
    if random.random() < px:
        img = np.fliplr(img).copy()
        if(len(labels) > 0):
            labels[:, 1] = height - labels[:, 1]
            labels[:, 3] = height - labels[:, 3]
            labels[:, [1, 3]] = labels[:, [3, 1]]
    return img, labels


def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    norm = (x - mean) / std
    """
    img = img / 255.0
    #mean = np.array(mean)
    #std = np.array(std)
    #img = (img - mean[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]
    return img.astype(np.float32)


def random_color_distort(src, brightness_delta=32, contrast_low=0.5, contrast_high=1.5,
                         saturation_low=0.5, saturation_high=1.5, hue_delta=18):
    """gluoncv/data/transforms/experimental/image.py
    Randomly distort image color space.
    Note that input image should in original range [0, 255].
    Parameters
    ----------
    src : numpy.ndarray
        Input image as HWC format.
    brightness_delta : int
        Maximum brightness delta. Defaults to 32.
    contrast_low : float
        Lowest contrast. Defaults to 0.5.
    contrast_high : float
        Highest contrast. Defaults to 1.5.
    saturation_low : float
        Lowest saturation. Defaults to 0.5.
    saturation_high : float
        Highest saturation. Defaults to 1.5.
    hue_delta : int
        Maximum hue delta. Defaults to 18.
    Returns
    -------
    numpy.ndarray
        Distorted image in HWC format.
    """
    def brightness(src, delta, p=0.5):
        """Brightness distortion."""
        if np.random.uniform(0, 1) > p:
            delta = np.random.uniform(-delta, delta)
            src += delta
            return src
        return src

    def contrast(src, low, high, p=0.5):
        """Contrast distortion"""
        if np.random.uniform(0, 1) > p:
            alpha = np.random.uniform(low, high)
            src *= alpha
            return src
        return src

    def saturation(src, low, high, p=0.5):
        """Saturation distortion."""
        if np.random.uniform(0, 1) > p:
            alpha = np.random.uniform(low, high)
            gray = src * np.array([[[0.299, 0.587, 0.114]]])
            gray = np.sum(gray, axis=2, keepdims=True)
            gray *= (1.0 - alpha)
            src *= alpha
            src += gray
            return src
        return src

    def hue(src, delta, p=0.5):
        """Hue distortion"""
        if np.random.uniform(0, 1) > p:
            alpha = random.uniform(-delta, delta)
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0],
                           [0.0, u, -w],
                           [0.0, w, u]])
            tyiq = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.274, -0.321],
                             [0.211, -0.523, 0.311]])
            ityiq = np.array([[1.0, 0.956, 0.621],
                              [1.0, -0.272, -0.647],
                              [1.0, -1.107, 1.705]])
            t = np.dot(np.dot(ityiq, bt), tyiq).T
            src = np.dot(src, t)
            return src
        return src

    src = src.astype(np.float32)
    # brightness
    src = brightness(src, brightness_delta)

    # color jitter
    if np.random.randint(0, 2):
        src = contrast(src, contrast_low, contrast_high)
        src = saturation(src, saturation_low, saturation_high)
        src = hue(src, hue_delta)
    else:
        src = saturation(src, saturation_low, saturation_high)
        src = hue(src, hue_delta)
        src = contrast(src, contrast_low, contrast_high)
    # return np.clip(src, 0, 255).astype(np.uint8)
    return src

def bbox_crop(labels, crop_box=None, allow_outside_center=True):
    """gluoncv code
    Crop bounding boxes according to slice area.
    This method is mainly used with image cropping to ensure bonding boxes fit
    within the cropped image.
    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    crop_box : tuple
        Tuple of length 4. :math:`(x_{min}, y_{min}, width, height)`
    allow_outside_center : bool
        If `False`, remove bounding boxes which have centers outside cropping area.
    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape (M, 4+) where M <= N.
    """
    bbox = labels[:, 1:].copy()
    if crop_box is None:
        return labels
    if not len(crop_box) == 4:
        raise ValueError(
            "Invalid crop_box parameter, requires length 4, given {}".format(str(crop_box)))
    if sum([int(c is None) for c in crop_box]) == 4:
        return labels

    l, t, w, h = crop_box

    left = l if l else 0
    top = t if t else 0
    right = left + (w if w else np.inf)
    bottom = top + (h if h else np.inf)
    crop_bbox = np.array((left, top, right, bottom))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        mask = np.logical_and(crop_bbox[:2] <= centers, centers < crop_bbox[2:]).all(axis=1)

    # transform borders
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bbox[:2])
    bbox[:, 2:4] = np.minimum(bbox[:, 2:4], crop_bbox[2:4])
    bbox[:, :2] -= crop_bbox[:2]
    bbox[:, 2:4] -= crop_bbox[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:4]).all(axis=1))
    labels = labels[mask]
    labels[:, 1:] = bbox[mask]
    return labels

def random_crop_with_constraints(bbox, size, min_scale=0.3, max_scale=1,
                                 max_aspect_ratio=2, constraints=None,
                                 max_trial=50):
    """gluoncv code
    Crop an image randomly with bounding box constraints.
    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_. More details can be found in
    data augmentation section of the original paper.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    size : tuple
        Tuple of length 2 of image shape as (width, height).
    min_scale : float
        The minimum ratio between a cropped region and the original image.
        The default value is :obj:`0.3`.
    max_scale : float
        The maximum ratio between a cropped region and the original image.
        The default value is :obj:`1`.
    max_aspect_ratio : float
        The maximum aspect ratio of cropped region.
        The default value is :obj:`2`.
    constraints : iterable of tuples
        An iterable of constraints.
        Each constraint should be :obj:`(min_iou, max_iou)` format.
        If means no constraint if set :obj:`min_iou` or :obj:`max_iou` to :obj:`None`.
        If this argument defaults to :obj:`None`, :obj:`((0.1, None), (0.3, None),
        (0.5, None), (0.7, None), (0.9, None), (None, 1))` will be used.
    max_trial : int
        Maximum number of trials for each constraint before exit no matter what.
    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape :obj:`(M, 4+)` where M <= N.
    tuple
        Tuple of length 4 as (x_offset, y_offset, new_width, new_height).
    """
    # default params in paper
    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    w, h = size

    candidates = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        min_iou = -np.inf if min_iou is None else min_iou
        max_iou = np.inf if max_iou is None else max_iou

        for _ in range(max_trial):
            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(h - crop_h)
            crop_l = random.randrange(w - crop_w)
            crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))

            if len(bbox) == 0:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                return bbox, (left, top, right-left, bottom-top)

            iou = iou_calc1(bbox[:, 1:], crop_bb[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                candidates.append((left, top, right-left, bottom-top))
                break

    # random select one
    while candidates:
        crop = candidates.pop(np.random.randint(0, len(candidates)))
        new_bbox = bbox_crop(bbox, crop, allow_outside_center=False)
        if new_bbox.size < 1:
            continue
        new_crop = (crop[0], crop[1], crop[2], crop[3])
        return new_bbox, new_crop
    return bbox, (0, 0, w, h)
