"""
Helper function for configuration
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
import shutil
from os.path import join

from yacs.config import CfgNode as CN



from util.create_dir import create_dirs
from util.setup_logging import setup_logging


def get_config():
    """
    Setup environment for running experiment

    Returns:
        config (CN object): the configuration for experiment
    """
    # parse command line arguments
    args_parser = argparse.ArgumentParser(description='Command Line Arguments')
    args_parser.add_argument('config', default='None',
                             help='The Configuration file in YAML format')
    args = args_parser.parse_args()

    # parse YAML config file
    config = CN(new_allowed=True)
    config.merge_from_file(args.config)

    # create directories
    config.env.summ_dir = join('exps', config.env.exp_name, 'summ')
    config.env.ckpt_dir = join('exps', config.env.exp_name, 'ckpt')
    config.env.out_dir = join('exps', config.env.exp_name, 'out')
    config.env.log_dir = join('exps', config.env.exp_name, 'log')
    create_dirs([config.env.summ_dir, config.env.ckpt_dir,
                 config.env.out_dir, config.env.log_dir])

    # setup logging
    setup_logging(config.env.log_dir)
    logger = logging.getLogger('Setup Configurations')
    logger.info('Parsed configuration file: "%s"', args.config)
    shutil.copyfile(args.config, join('exps', config.env.exp_name,
                                      'config.yaml'))
    logging.info(config)

    # welcome message
    try:
        logger.info('The experiment: %s will start now', config.env.exp_name)
    except AttributeError:
        logger.warning('config.env.exp_name not specified')

    return config

