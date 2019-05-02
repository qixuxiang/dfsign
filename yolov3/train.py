import argparse
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler

import test  # Import test.py to get mAP
from models import *
from utils.utils import *
from data.dataset_voc import VOCDetection
from data.dataset_tt100k import TT100KDetection

from pprint import pprint
import pdb

def train(
        cfg,
        img_size=416,
        resume=False,
        epochs=273,  # 500200 batches at bs 64, dataset length 117263
        batch_size=16,
        accumulate=1,
        multi_scale=False,
        freeze_backbone=False,
        num_workers=4,
        transfer=False  # Transfer learning (train only YOLO layers)

):
    weights = 'weights' + os.sep
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device()

    if multi_scale:
        img_size = 608  # initiate with maximum multi_scale size
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Initialize model
    model = Darknet(cfg, img_size).to(device)
    
    # Optimizer
    lr0 = 0.001  # initial learning rate
    optimizer = SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=0.0005)
    
    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    yl = get_yolo_layers(model)  # yolo layers
    nf = int(model.module_defs[yl[0] - 1]['filters'])  # yolo layer size (i.e. 255)

    if resume:  # Load previously saved model
        if transfer:  # Transfer learning
            chkpt = torch.load(weights + 'yolov3-spp.pt', map_location=device)
            model.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255},
                                  strict=False)
            for p in model.parameters():
                p.requires_grad = True if p.shape[0] == nf else False
        else:  # resume from latest.pt
            chkpt = torch.load(latest, map_location=device)  # load checkpoint
            model.load_state_dict(chkpt['model'])
        
        start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_loss = chkpt['best_loss']
        del chkpt
    else:  # Initialize model with backbone (optional)
        if '-tiny.cfg' in cfg:
            cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
        else:
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')

    # multi gpus
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # Set scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 90], gamma=0.1, last_epoch=start_epoch - 1)

    # Dataset
    # train_dataset = VOCDetection(root=os.path.join('~', 'data', 'VOCdevkit'), img_size=img_size, mode='train')
    train_dataset = TT100KDetection(root=os.path.join('~', 'data', 'TT100K', 'TT100K_chip_voc'), 
                                    img_size=img_size, mode='train')

    # Dataloader
    dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=train_dataset.collate_fn)

    # Start training
    t = time.time()
    # model_info(model)
    nB = len(dataloader)
    n_burnin = nB  # burn-in batches
    for epoch in range(start_epoch, epochs):
        model.train()
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        mloss = defaultdict(float)  # mean loss

        for i, (imgs, targets, _, paths) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            nt = len(targets)
            if nt == 0:  # if no targets continue
                continue

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = lr0 * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            optimizer.zero_grad()
            # Run model
            pred = model(imgs)
            # Build targets
            target_list = build_targets(model, targets)
            # Compute loss
            loss, loss_dict = compute_loss(pred, target_list)
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nB:
                optimizer.step()

            # Running epoch-means of tracked metrics
            for key, val in loss_dict.items():
                mloss[key] = (mloss[key] * i + val) / (i + 1)

            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nB - 1),
                mloss['xy'], mloss['wh'], mloss['conf'], mloss['cls'],
                mloss['total'], nt, time.time() - t)
            t = time.time()
            if i % 30 == 0:
                print(s)

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                dataset.img_size = random.choice(range(10, 20)) * 32
                print('multi_scale img_size = %g' % dataset.img_size)

        # Update best loss
        if mloss['total'] < best_loss:
            best_loss = mloss['total']

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                    'best_loss': best_loss,
                    'model': model.module.state_dict() if type(model) is nn.parallel.DataParallel else model.state_dict(),
                    'optimizer': optimizer.state_dict()}
        if epoch % 5 == 0:
            torch.save(checkpoint, 'weights/epoch_tt100k_%03d.pt' % epoch)

        # if epoch > 9 and epoch % 10 == 0:
        if False:
            with torch.no_grad():
                APs, mAP = test.test(cfg, weights=None, batch_size=32, img_size=img_size, model=model)
                pprint(APs)
                print(mAP)

        del checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=416, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    train(
        opt.cfg,
        img_size=opt.img_size,
        resume=opt.resume or opt.transfer,
        transfer=opt.transfer,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        multi_scale=opt.multi_scale,
        num_workers=opt.num_workers
    )
