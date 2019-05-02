import os
import torch
from utils.dataset_voc import VOCDetection
import pdb

voc = VOCDetection(root=os.path.join('~', 'data', 'VOCdevkit'), img_size=416)

dataloader = torch.utils.data.DataLoader(voc, batch_size=1,
    num_workers=1, pin_memory=False)

for imgs, targets, shapes, _ in dataloader:
    print(targets)
