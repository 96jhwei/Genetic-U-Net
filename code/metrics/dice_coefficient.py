"""
Dice score coefficient
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


def hard_dice(input_, target, threshold=0.5, reduction='mean', epsilon=1e-8):
    """
    Hard dice score coefficient after thresholding.

    Arguments:
        preds (torch tensor): raw probability outputs
        targets (torch tensor): ground truth
        threshold (float): threshold value, default: 0.5
        reduction (string): one of 'none', 'mean' or 'sum'
        epsilon (float): epsilon for numerical stability, default: 1e-8

    Returns:
        dice (torch tensor): hard dice score coefficient
    """
    if not input_.shape == target.shape:
        raise ValueError

    # if not (input_.max() <= 1.0 and input_.min() >= 0.0):
    #     raise ValueError

    if not ((target.max() == 1.0 and target.min() == 0.0 and(target.unique().numel() == 2)) 
        or (target.max() == 0.0 and target.min() == 0.0 and(target.unique().numel() == 1))):
        raise ValueError

    input_threshed = input_.clone()
    input_threshed[input_ < threshold] = 0.0
    input_threshed[input_ >= threshold] = 1.0

    intesection = torch.sum(input_threshed * target, dim=-1)
    input_norm = torch.sum(input_threshed, dim=-1)
    target_norm = torch.sum(target, dim=-1)
    dice = torch.div(2.0 * intesection + epsilon,
                     input_norm + target_norm + epsilon)

    if reduction == 'none':
        pass
    elif reduction == 'mean':
        dice = torch.mean(dice)
    elif reduction == 'sum':
        dice = torch.sum(dice)
    else:
        raise NotImplementedError

    return dice


class HardDice(nn.Module):
    """
    Hard dice module
    """

    def __init__(self, threshold=0.5, reduction='mean'):
        super(HardDice, self).__init__()
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, input_, target):
        dice = hard_dice(input_=input_, target=target,
                         threshold=self.threshold,
                         reduction=self.reduction,
                         epsilon=1e-8)
        return dice
