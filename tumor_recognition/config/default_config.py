#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : ZhenHuang (hz13@mail.ustc.edu.cn)
Date               : 2021-12-21 16:50
Last Modified By   : ZhenHuang (hz13@mail.ustc.edu.cn)
Last Modified Date : 2021-12-26 16:57
Description        : Default Configurations
-------- 
Copyright (c) 2021 Multimedia Group USTC. 
'''
from yacs.config import CfgNode
from typing import Optional, Union, Callable, Dict, List, Tuple


def _assert_in(opt, opts):
    assert (opt in opts), '{} not in options {}'.format(opt, opts)
    return opt


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()
_C.ROOT_LOG = 'log'  # log dir name
_C.ROOT_MODEL = 'checkpoint'  # model dir name
_C.MODEL_SUFFIX = ''  # model suffix mark
_C.PRINT_FREQ = 20  # print every $PRINT_FREQ$ steps.
_C.EVAL_FREQ = 1  # print every $EVAL_FREQ$ epochs.
_C.NUM_GPUS = 1
_C.DIST = False  # Use DistributedDataParallel (DDP)
_C.DIST_BN = 'reduce'
_C.SEED = 3407
_C.SHARD_ID = 0
_C.NUM_SHARDS = 2
_C.INIT_METHOD = 'tcp://localhost:9997'
_C.DEVICE = 'cpu'
_C.EVAL_CHECKPOINT = ''
_C.EVAL_METRIC = 'acc'
_C.EXP_NAME = 'DEBUG'
_C.OUTPUT_DIR = '/Users/zhushuhan/Desktop/Medical-zhen/output/'
_C.CHECKPOINT_HISTORY = 10
_C.RECOVERY_FREQ = 10000

# -----------------------------------------------------------------------------
# DATA options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()
_C.DATA.DIR = '/Users/zhushuhan/Desktop//med_epe-Ax-T1_T1_E_T2.hdf5'
_C.DATA.BATCH_SIZE = 16
_C.DATA.TEST_BATCH_SIZE = 64
_C.DATA.NUM_WORKERS = 4
# _C.DATA.MEAN = [0.485, 0.456, 0.406]
# _C.DATA.STD = [0.229, 0.224, 0.225]
_C.DATA.TEST_RATIO = 0.2
_C.DATA.NUM_FRAMES = 6
_C.DATA.SAMPLE_STEP = 5
_C.DATA.SEED = 1
_C.DATA.MODALITY = ['T1_Ax_reg', 'T1_E_Ax_reg', 'T2_Ax_reg']

# -----------------------------------------------------------------------------
# TRAIN options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()
# optimizer
_C.TRAIN.OPTIM = 'adam'
_C.TRAIN.BASE_LR = 0.001
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.CLIP_GRADIENT = 20.0
_C.TRAIN.WORKERS = 8
_C.TRAIN.TUNING = False
_C.TRAIN.FREEZE_BN = False
_C.TRAIN.SYNC_BN = True
_C.TRAIN.TUNE_FROM = ''

# save & load
_C.TRAIN.RESUME = ''
_C.TRAIN.RESUME_OPTIM = True

# scheduler
_C.TRAIN.EPOCHS = 50
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.SCHED = 'step'
_C.TRAIN.MIN_LR = 1e-5
_C.TRAIN.DECAY_RATE = 0.1
_C.TRAIN.DECAY_EPOCHS = 300
_C.TRAIN.DECAY_EPOCHS_LIST = []
_C.TRAIN.WARMUP_LR = 0.0001
_C.TRAIN.WARMUP_EPOCHS = 0

# -----------------------------------------------------------------------------
# MODEL options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.NAME = 'resnet20'
_C.MODEL.NUM_BLOCKS = [3, 3, 3]
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.IN_CHANNELS = 3

# -----------------------------------------------------------------------------
# DEBUG options
# -----------------------------------------------------------------------------
_C.DEBUG = CfgNode()
_C.DEBUG.VIS_USE_TB = True


def _assert_and_infer_cfg(cfg):
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())