
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


class FocalLossForSigmoid(nn.Module):
    def __init__(self, gamma=2, alpha=0.55, reduction='mean'):
        super(FocalLossForSigmoid, self).__init__()
        self.gamma = gamma
        assert 0 <= alpha <= 1, 'The value of alpha must in [0,1]'
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCELoss(reduce=False)

    def forward(self, input_, target):
        input_ = torch.clamp(input_, min=1e-7, max=(1 - 1e-7))

        if self.alpha != None:
            loss = (self.alpha * target + (1 - target) * (1 - self.alpha)) * (
                torch.pow(torch.abs(target - input_), self.gamma)) * self.bce(input_, target)
        else:
            loss = torch.pow(torch.abs(target - input_), self.gamma) * self.bce(input_, target)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            pass

        return loss


