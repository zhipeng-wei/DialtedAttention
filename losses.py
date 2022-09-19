import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(LogitLoss, self).__init__()
        self.reduction = reduction
    def forward(self, logits, labels):
        real = logits.gather(1,labels.unsqueeze(1)).squeeze(1)
        logit_dists = ( -1 * real)
        if self.reduction == 'none':
            return logit_dists
        else:
            loss = logit_dists.mean()
            return loss

