import argparse
import json
import time
import shutil
from tqdm import tqdm
from pathlib import Path

from models import *
from data.dataset_voc import VOCDetection
from utils.utils import *
from eval.voc_eval import *
import pdb

def test(
        cfg,
        weights,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.001,
        nms_thres=0.5,
        model=None
):
    if model is None:
        device = torch_utils.select_device()

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'], strict=False)
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

    else:
        device = next(model.parameters()).device  # get model device

    # Get dataloader
    vocset = VOCDetection(root=os.path.join('~', 'data', 'VOCdevkit'), splits=((2007, 'test'),),
                        img_size=img_size, mode='test')
    dataloader = torch.utils.data.DataLoader(vocset, 
                                            batch_size=batch_size, 
                                            num_workers=8,
                                            collate_fn=vocset.collate_fn)

    nC = vocset.num_class #num class
    classes = vocset.classes

    det_results_path = os.path.join('eval', 'results', 'VOC2007', 'Main')
    if os.path.exists(det_results_path):
        shutil.rmtree(det_results_path)
    os.makedirs(det_results_path)

    model.eval()
    seen = 0
    pbar = tqdm(total=len(dataloader) * batch_size, desc='Computing mAP')
    for batch_i, (imgs, targets, shapes, img_paths) in enumerate(dataloader):
        output, _ = model(imgs.to(device))
        # nms
        output = nms(output, conf_thres, nms_thres, method='nms')

        for si, detections in enumerate(output):
            seen += 1
            if len(detections) == 0:
                continue

            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], shapes[si]).round()

            image_ind = os.path.split(img_paths[si])[-1][:-4]
            for bbox in detections:
                coor = bbox[:4]
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = classes[class_ind]
                score = score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                bbox_mess = ' '.join([image_ind, score, xmin, ymin, xmax, ymax]) + '\n'
                with open(os.path.join(det_results_path, 'comp3_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(bbox_mess)
        
        pbar.update(batch_size)
    pbar.close()

    filename = os.path.join('eval', 'results', 'VOC2007', 'Main', 'comp3_det_test_{:s}.txt')
    cachedir = os.path.join('eval', 'cache')
    annopath = os.path.join(vocset._root, 'VOC2007', 'Annotations', '{:s}.xml')
    imagesetfile = os.path.join(vocset._root, 'VOC2007', 'ImageSets', 'Main', 'test.txt')
    APs = {}
    for i, cls in enumerate(classes):
        rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thres, False)
        APs[cls] = ap
    # if os.path.exists(cachedir):
    #     shutil.rmtree(cachedir)
    mAP = np.mean([APs[cls] for cls in APs])
    return APs, mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-voc.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    opt = parser.parse_args()
    print(opt, end='\n\n')
    
    with torch.no_grad():
        APs, mAP = test(
            opt.cfg,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.iou_thres,
            opt.conf_thres,
            opt.nms_thres)
        print(APs, mAP)
