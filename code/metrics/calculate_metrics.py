import sys

sys.path.append('../')
from metrics.binary_confusion_matrix import get_binary_confusion_matrix, get_threshold_binary_confusion_matrix
from metrics.binary_statistical_metrics import get_accuracy, get_true_positive_rate, get_true_negative_rate, \
    get_precision, get_f1_socre, get_iou
from metrics.dice_coefficient import hard_dice
from metrics.pr_curve import get_pr_curve
from metrics.roc_curve import get_auroc, get_roc_curve
from util.numpy_utils import tensor2numpy
import numpy as np
from copy import deepcopy

# np.seterr(divide='ignore', invalid='ignore')


def calculate_metrics( preds, targets, device,config=None):
    curr_TP, curr_FP, curr_TN, curr_FN = get_binary_confusion_matrix(
        input_=preds, target=targets, device=device, pixel=0,
        threshold=0.5,
        reduction='sum')

    curr_acc = get_accuracy(true_positive=curr_TP,
                            false_positive=curr_FP,
                            true_negative=curr_TN,
                            false_negative=curr_FN)

    curr_recall = get_true_positive_rate(true_positive=curr_TP,
                                         false_negative=curr_FN)

    curr_specificity = get_true_negative_rate(false_positive=curr_FP,
                                              true_negative=curr_TN)

    curr_precision = get_precision(true_positive=curr_TP,
                                   false_positive=curr_FP)

    curr_f1_score = get_f1_socre(true_positive=curr_TP,
                                 false_positive=curr_FP,
                                 false_negative=curr_FN)

    curr_iou = get_iou(true_positive=curr_TP,
                       false_positive=curr_FP,
                       false_negative=curr_FN)

    curr_auroc = get_auroc(preds, targets)

    return (curr_acc, curr_recall, curr_specificity, curr_precision,
            curr_f1_score, curr_iou, curr_auroc)

