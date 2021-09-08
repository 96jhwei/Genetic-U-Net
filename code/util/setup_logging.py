"""
Helper function for setup logging
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from logging.handlers import RotatingFileHandler

from os.path import join


def setup_logging(log_dir):
    """
    Setup logging

    Args:
        log_dir (string): the directory, to which the logging file is saved
    """
    log_file_format = '[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s'
    log_console_format = '[%(levelname)s]: %(message)s'

    # main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler(
        join(log_dir, 'exp_debug.log'), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(logging.Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(
        join(log_dir, 'exp_error.log'), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(logging.Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)
