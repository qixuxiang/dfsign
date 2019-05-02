"""
Focal Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class FocalLoss(nn.modules.loss._Loss):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self._gamma = gamma
        self._alpha = alpha
        self._eps = 1e-12
        self.reduction = reduction

    def forward(self, input, target, weight=None):
        one_hot = target > 0
        pt = torch.where(one_hot, input, 1-input)
        t = torch.ones_like(one_hot, dtype=torch.float)
        alpha = torch.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        logpt = torch.log(torch.min(pt + self._eps, t))
        
        loss = -1 * alpha * (1 - pt) ** self._gamma * logpt
        if not weight is None:
            loss *= weight
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()