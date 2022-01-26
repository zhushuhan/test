#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : ZhenHuang (hz13@mail.ustc.edu.cn)
Date               : 2021-12-21 21:08
Last Modified By   : ZhenHuang (hz13@mail.ustc.edu.cn)
Last Modified Date : 2021-12-22 13:06
Description        : Visualization
-------- 
Copyright (c) 2021 Multimedia Group USTC. 
'''
import os
import pdb
import torch
import logging
import torch.nn.functional as F
import torchvision
import numpy as np
from typing import Union, Optional, List, Tuple, Text, BinaryIO
from torch.utils.tensorboard import SummaryWriter

_logger = logging.getLogger('vis')

class Visualizer(object):
    def __init__(self, model=None, train_loader=None, test_loader=None, use_tb=True, root='./vis', device='cuda'):
        if model is not None and getattr(model, 'module') is not None:
            model = model.module
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.use_tb = use_tb
        if use_tb:
            self.tb_writer = SummaryWriter(log_dir=root)
        self.root = root
        self.device = device


    def visualize(self, epoch, train_metrics={}, eval_metrics={}):
        if self.use_tb:
            self.update_tensorboard(epoch, train_metrics, eval_metrics)

    def update_tensorboard(self, epoch, train_metrics={}, eval_metrics={}):
        for k, v in train_metrics.items():
            self.tb_writer.add_scalar('train/{}'.format(k), v, epoch)
        for k, v in eval_metrics.items():
            self.tb_writer.add_scalar('eval/{}'.format(k), v, epoch)

