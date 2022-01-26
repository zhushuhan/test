#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : ZhenHuang (hz13@mail.ustc.edu.cn)
Date               : 2021-12-23 15:02
Last Modified By   : ZhenHuang (hz13@mail.ustc.edu.cn)
Last Modified Date : 2021-12-24 12:47
Description        : Temporal Shift Model (2D Model)
-------- 
Copyright (c) 2021 Multimedia Group USTC. 

Reference
---------
https://github.com/mit-han-lab/temporal-shift-module
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models.resnet import BasicBlock, Bottleneck

from typing import Union, List

from tumor_recognition.models.resnet_cifar import BasicBlock


def TSM_ResNet(nn.Module):
    def __init__(
        self,
        block: Union[BasicBlock, Bottleneck],
        num_blocks: List[int],
        in_channels: int = 3,
        out_channels: int = 1000,
        non_local: bool = False,
    ):
    super().__init__()
    return