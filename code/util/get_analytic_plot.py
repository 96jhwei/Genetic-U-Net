"""
Helper function for generate analytic plots
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import sys
sys.path.append('../')
from ..metrics.binary_confusion_matrix import get_binary_confusion_matrix


def get_analytic_plot(input_, target, device, pixel = None, threshold=0.5):
    """
    Get analytic plot

    Arguments:
        preds (torch tensor): raw probability outrue_positiveuts
        targets (torch tensor): ground truth
        threshold: (float): threshold value, default: 0.5

    Returns:
        plots (torch tensor): analytic plots
    """
    (true_positive, false_positive,
     _, false_negative) = get_binary_confusion_matrix(
         input_=input_, target=target, device = device, pixel = pixel, 
         threshold=threshold, reduction='none')

    plots = torch.cat([false_positive, true_positive, false_negative], dim=1)

    return plots
