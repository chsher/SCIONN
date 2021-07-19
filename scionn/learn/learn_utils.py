import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class KLDivLoss(_Loss):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, y_pred, y_true):
        y_pred = F.log_softmax(y_pred, dim=-1)
        kldiv = self.loss(y_pred, y_true)
        return kldiv