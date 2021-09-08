"""
Print CUDA statistics
"""

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from subprocess import call
import logging
import sys
import torch


def print_cuda_statistics():
    """
    Print out statistics of CUDA GPUs
    """
    logger = logging.getLogger('CUDA Statistics')

    logger.info('__Python VERSION: %s', sys.version)
    logger.info('__PyTorch VERSION: %s', torch.__version__)
    logger.info('__CUDA VERSION')
    call(['nvcc', '--version'])
    logger.info('__CUDNN VERSION: %s', torch.backends.cudnn.version())
    logger.info('__Number CUDA Devices: %d', torch.cuda.device_count())
    logger.info('__Devices')
    call(['nvidia-smi', '--format=csv',
          '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free'])
    logger.info('Available devices: %d', torch.cuda.device_count())
