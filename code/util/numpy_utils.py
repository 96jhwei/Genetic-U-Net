"""
Utilities for using numpy and sklearn with PyTorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def tensor2float(input_):
    """
    Convert a one element tensor into float
    """
    if isinstance(input_, torch.Tensor):
        if input_.numel() == 1:
            output = input_.item()
        else:
            raise ValueError

    return output


def tensor2numpy(input_):
    """
    Convert a torch tensor to numpy array
    """
    if isinstance(input_, torch.Tensor):
        try:
            output = input_.numpy()

        except RuntimeError:
            try:
                output = input_.detach().numpy()
            except TypeError:
                output = input_.detach().cpu().numpy()

        except TypeError:
            try:
                output = input_.cpu().numpy()
            except RuntimeError:
                output = input_.cpu().detach().numpy()
    else:
        pass

    return output


def flatten_tensor(input_):
    """
    Flatten PyTorch tensor input into numpy ndarray for using numpy and
         sklearn metric functions.

    Arguments:
        input_: torch tensor of arbitrary shape

    Returns:
        output: flattened numpy array
    """
    output = tensor2numpy(input_)

    return output.flatten()
