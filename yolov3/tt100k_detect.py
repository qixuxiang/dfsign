import argparse
import time
import json
import shutil
from tqdm import tqdm
from sys import platform

from models import *
from data.datasets import *
from utils.utils import *
from data.dataset_tt100k import TT100KDetection

import pdb

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def detect(
        cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        webcam=False
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    model.to(device).eval()

    tt100k = True
    # Set Dataloader
    if tt100k:
        root_dir = '~/data/TT100K/TT100K_chip_voc'
        root_dir = os.path.expanduser(root_dir)
        list_file = os.path.join(root_dir, 'ImageSets/Main/test.txt')
        image_dir = os.path.join(root_dir, 'JPEGImages')
        with open(list_file, 'r') as f:
            images = [x.strip() for x in f.readlines()]
        images = [os.path.join(image_dir, x+'.jpg') for x in images]

    dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = TT100KDetection.CLASSES
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    save_images = False
    results = dict()
    for i, (path, img, im0, vid_cap) in enumerate(tqdm(dataloader)):
        t = time.time()
        im_id = path.split('/')[-1][:-4]
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        detections = nms(pred, conf_thres, nms_thres, method='nms')[0]
        detections = detections[detections[:, 4] > 0.6]

        if detections is not None and len(detections) > 0:
            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            results[im_id] = {'pred_box': detections[:,:4],
                           'pred_score': detections[:,4],
                           'pred_label': [classes[int(i)] for i in detections[:,5]]}
            
            # Add bbox to the image
            if save_images:
                # Print results to screen
                for c in np.unique(detections[:, -1]):
                    n = (detections[:, -1] == c).sum()
                    print('%g %ss' % (n, classes[int(c)]), end=', ')
                for *xyxy, conf, cls in detections:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
        if save_images:
            cv2.imwrite(save_path, im0)
            print('Done. (%.3fs)' % (time.time() - t))

    with open(output + '/results.json', 'w') as f:
        json.dump(results, f, cls=MyEncoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tt100k.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )
