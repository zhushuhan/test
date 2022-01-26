#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : ZhenHuang (hz13@mail.ustc.edu.cn)
Date               : 2021-12-21 16:45
Last Modified By   : ZhenHuang (hz13@mail.ustc.edu.cn)
Last Modified Date : 2021-12-21 16:45
Description        : Common used functions
-------- 
Copyright (c) 2021 Multimedia Group USTC. 
'''

from __future__ import absolute_import
import os
import sys
import errno
import json
import os.path as osp
import shutil
import argparse
import torch.distributed as dist
from typing import List

import torch


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(
            type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state,
                    is_best,
                    fpath='checkpoint.pth.tar',
                    model_path='model_1_best.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), model_path))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def run(
    local_rank,
    num_proc,
    func,
    init_method,
    shard_id,
    num_shards,
    backend,
    cfg,
    output_queue=None,
):
    """
    @Source: PySlowFast
    for DistributedDataParallel (DDP)
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        num_proc (int): number of processes per machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        shard_id (int): the rank of the current machine.
        num_shards (int): number of overall machines for the distributed
            training job.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        output_queue (queue): can optionally be used to return values from the
            master process.
    """

    # the correct local_rank can be obtained in this function
    cfg.LOCAL_RANK = local_rank
    cfg.DEVICE = 'cuda:{}'.format(cfg.LOCAL_RANK)

    # Initialize the process group.
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank

    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        raise e

    torch.cuda.set_device(local_rank)
    ret = func(cfg)
    if output_queue is not None and local_rank == 0:
        output_queue.put(ret)


def check_rootfolders(cfg):
    """Create log and model folder"""
    folders_util = [cfg.ROOT_LOG, cfg.ROOT_MODEL]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


def save_cfg(cfg):
    model_dir = '%s/%s' % (cfg.ROOT_MODEL, cfg.STORE_NAME)
    if os.path.exists(model_dir) is False:
        os.mkdir(model_dir)
    filename = '%s/config.yaml' % (model_dir)
    with open(filename, 'w') as f:
        f.write(cfg.dump())


def is_master_rank(cfg):
    """The global rank is 0 or not.

    Parameters
    ----------
    cfg : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if cfg.DIST:
        return dist.get_rank() == 0
    else:
        return True



def create_scheduler(optimizer,
                     epochs: int,
                     sched: str = 'step',
                     min_lr: float = 0,
                     decay_rate: float = 0.1,
                     decay_epochs: int = 30,
                     decay_epochs_list: List = [],
                     warmup_lr: float = 0,
                     warmup_epochs: int = 0,
                     **kwargs):
    from timm.scheduler import create_scheduler
    args = argparse.Namespace()
    args.epochs = epochs
    args.sched = sched
    args.min_lr = min_lr
    args.decay_rate = decay_rate
    args.decay_epochs = decay_epochs if sched in ['step'] else decay_epochs_list
    args.warmup_lr = warmup_lr
    args.warmup_epochs = warmup_epochs
    option_arg_names = [
        'lr_noise', 'lr_cycle_mul', 'lr_cycle_limit', 'lr_noise_pct',
        'lr_noise_std', 'seed'
    ]
    for name in kwargs:
        if name in option_arg_names:
            setattr(args, name, kwargs[name])

    return create_scheduler(args, optimizer)
