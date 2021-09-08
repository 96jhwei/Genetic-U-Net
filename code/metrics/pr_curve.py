"""
Precision recall curve and area under it
"""

from sklearn.metrics import precision_recall_curve
import sys
sys.path.append('../')
from util.numpy_utils import flatten_tensor


def get_pr_curve(preds, targets):
    """
    Get precision recall curve

    Arguments:
        preds(torch tensor): raw probability outputs
        targets(torch tensor): ground truth

    Returns:
        precisions
        recalls
        thresholds
    """
    preds, targets = list(map(flatten_tensor, [preds, targets]))
    precisions, recalls, thresholds = precision_recall_curve(
        y_true=targets,
        probas_pred=preds,
        pos_label=None,
        sample_weight=None
    )

    return precisions, recalls, thresholds
