#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : ZhenHuang (hz13@mail.ustc.edu.cn)
Date               : 2021-12-21 16:28
Last Modified By   : ZhenHuang (hz13@mail.ustc.edu.cn)
Last Modified Date : 2021-12-26 16:57
Description        : Main training pipeline
-------- 
Copyright (c) 2021 Multimedia Group USTC. 
'''

from __future__ import print_function, absolute_import
import os
import sys
import pdb
import argparse
import operator
import logging
import numpy as np
import os.path as osp
from datetime import datetime
import time
from collections import OrderedDict
from typing import Optional, Callable, Dict, List, Tuple

# torch
import torch
from torch import optim
import torchvision
import torch.distributed as dist
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from yacs.config import CfgNode

# custom
import utils
from models import create_model
from utils import create_scheduler, AverageMeter
from config.default_config import get_cfg
from dataset import get_brain_hdf5_dataset_v1
from vis import Visualizer

# timm
import timm
from timm.utils import setup_default_logging, get_outdir, CheckpointSaver, update_summary, accuracy, random_seed
from timm.optim import create_optimizer_v2 as create_optimizer
from timm.models import load_checkpoint, resume_checkpoint, safe_model_name, model_parameters

# from vis import Visualizer

has_apex = False
# try:
#     from apex import amp
#     from apex.parallel import DistributedDataParallel as ApexDDP
#     from apex.parallel import convert_syncbn_model

#     has_apex = True
# except ImportError:
#     pass

has_native_amp = False
# try:
#     if getattr(torch.cuda.amp, 'autocast') is not None:
#         has_native_amp = True
# except AttributeError:
#     pass

_logger = logging.getLogger('train')

best_prec1 = 0


def get_data(cfg):
    train_set, test_set = get_brain_hdf5_dataset_v1(
        cfg.DIR,
        modality=cfg.MODALITY,
        test_ratio=cfg.TEST_RATIO,
        num_frames=cfg.NUM_FRAMES,
        sample_step=cfg.SAMPLE_STEP,
        seed=cfg.SEED)

    trainloader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=cfg.BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=cfg.NUM_WORKERS)
    testloader = torch.utils.data.DataLoader(dataset=test_set,
                                             batch_size=cfg.TEST_BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=cfg.NUM_WORKERS)
    return trainloader, testloader


def main(cfg: CfgNode):
    """Get default settings"""
    cudnn.benchmark = True
    rank = dist.get_rank() if cfg.DIST else 0
    random_seed(cfg.SEED, rank)
    if cfg.EXP_NAME:
        exp_name = '-'.join([
            cfg.EXP_NAME,
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        ])
    else:
        exp_name = '-'.join([
            safe_model_name(cfg.MODEL.NAME),
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        ])
    output_dir = get_outdir(
        cfg.OUTPUT_DIR if cfg.OUTPUT_DIR else './output/train', exp_name)
    setup_default_logging(log_path=osp.join(output_dir, 'log.txt'))
    # global best_prec1
    if utils.is_master_rank(cfg):
        _logger.info(cfg)
    _logger.info('is_master_rank(cfg): {}'.format(utils.is_master_rank(cfg)))

    """Build data loader"""
    train_loader, test_loader = get_data(cfg.DATA)

    """Build Models"""
    model = create_model(cfg.MODEL.NAME, cfg.MODEL).to(cfg.DEVICE)
    if cfg.DIST:
        if cfg.TRAIN.SYNC_BN:
            logging.info('Apply the Sync BatchNorm')
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(cfg.DEVICE)

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.LOCAL_RANK])
    else:
        model = torch.nn.DataParallel(model)
    if utils.is_master_rank(cfg):
        # _logger.info(model)
        _logger.info(
            'Model %s created, param count: %d' %
            (cfg.MODEL.NAME, sum([m.numel() for m in model.parameters()])))
    #TODO: model apex in timm

    """Build Optimizers"""
    optimizer = create_optimizer(model, cfg.TRAIN.OPTIM, cfg.TRAIN.BASE_LR,
                                 cfg.TRAIN.WEIGHT_DECAY)

    """Resume Models"""
    resume_epoch = None
    if cfg.TRAIN.RESUME:
        resume_epoch = resume_checkpoint(
            model,
            cfg.TRAIN.RESUME,
            optimizer=optimizer if cfg.TRAIN.RESUME_OPTIM else None,
            loss_scaler=None,
            log_info=cfg.LOCAL_RANK == 0)

    """Define loss function (criterion)"""
    train_loss_fn = nn.CrossEntropyLoss()
    validate_loss_fn = nn.CrossEntropyLoss()
    #train_loss_fn = nn.CrossEntropyLoss().cuda()
    #validate_loss_fn = nn.CrossEntropyLoss().cuda()

    """Validate"""
    if cfg.EVAL_CHECKPOINT:
        load_checkpoint(model, cfg.EVAL_CHECKPOINT, use_ema=False)
        val_metrics = validate(model, test_loader, validate_loss_fn, cfg)
        _logger.info("Top-1 accuracy of the model is: {:.2f}".format(
            val_metrics['acc']))
        return

    """Setup Logger, Saver and Visualizer"""
    eval_metric = cfg.EVAL_METRIC
    best_metric = None
    best_epoch = None
    saver = None
    viser = None
    if utils.is_master_rank(cfg):
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(model=model,
                                optimizer=optimizer,
                                args=None,
                                model_ema=None,
                                amp_scaler=None,
                                checkpoint_dir=output_dir,
                                recovery_dir=output_dir,
                                decreasing=decreasing,
                                max_history=cfg.CHECKPOINT_HISTORY)
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            f.write(cfg.dump())
        viser = Visualizer(use_tb=cfg.DEBUG.VIS_USE_TB, root=output_dir)

    """Scheduler"""
    lr_scheduler, num_epochs = create_scheduler(
        optimizer,
        cfg.TRAIN.EPOCHS,
        sched=cfg.TRAIN.SCHED,
        min_lr=cfg.TRAIN.MIN_LR,
        decay_rate=cfg.TRAIN.DECAY_RATE,
        decay_epochs=cfg.TRAIN.DECAY_EPOCHS,
        decay_epochs_list=cfg.TRAIN.DECAY_EPOCHS_LIST,
        warmup_lr=cfg.TRAIN.WARMUP_LR,
        warmup_epochs=cfg.TRAIN.WARMUP_EPOCHS)
    start_epoch = 0
    if cfg.TRAIN.START_EPOCH is not None:
        start_epoch = cfg.TRAIN.START_EPOCH
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)
    if utils.is_master_rank(cfg):
        _logger.info('Scheduled epochs: {}'.format(num_epochs))
        
    """Training Loops"""
    try:
        for epoch in range(start_epoch, num_epochs):
            train_metrics = train_one_epoch(epoch,
                                            model,
                                            train_loader,
                                            optimizer,
                                            train_loss_fn,
                                            cfg,
                                            lr_scheduler=lr_scheduler,
                                            saver=saver,
                                            output_dir=output_dir)
            eval_metrics = validate(model, test_loader, validate_loss_fn, cfg)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(epoch,
                               train_metrics,
                               eval_metrics,
                               os.path.join(output_dir, 'summary.csv'),
                               write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    epoch, metric=save_metric)

            if viser is not None:
                viser.visualize(epoch, train_metrics, eval_metrics)
    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(
            best_metric, best_epoch))


def validate(model, loader, loss_fn, cfg) -> OrderedDict:
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    accs_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            #input = input.cuda()
            #target = target.cuda()

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc = accuracy(output, target)[0]

            #torch.cuda.synchronize()

            losses_m.update(loss.data.item(), input.size(0))
            accs_m.update(acc.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_master_rank(cfg) and (last_batch or
                                              batch_idx % cfg.PRINT_FREQ == 0):
                log_name = 'Test'
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc: {acc.val:>7.4f} ({acc.avg:>7.4f})'.format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        acc=accs_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('acc', accs_m.avg)])

    return metrics


def train_one_epoch(epoch,
                    model,
                    loader,
                    optimizer,
                    loss_fn,
                    cfg,
                    lr_scheduler=None,
                    saver=None,
                    output_dir=None,
                    loss_scaler=None,
                    model_ema=None,
                    mixup_fn=None) -> OrderedDict:
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    accs_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        #input, target = input.cuda(), target.cuda()
        output = model(input)
        loss = loss_fn(output, target)
        acc = accuracy(output, target)[0]
        if not cfg.DIST:
            losses_m.update(loss.item(), input.size(0))
            accs_m.update(acc.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        if cfg.TRAIN.CLIP_GRADIENT is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg.TRAIN.CLIP_GRADIENT)
        optimizer.step()

        #torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % cfg.PRINT_FREQ == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            if utils.is_master_rank(cfg):
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Acc: {acc.val:>7.4f} ({acc.avg:>7.4f})  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx,
                        len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        acc=accs_m,
                        batch_time=batch_time_m,
                        lr=lr,
                        data_time=data_time_m))
        if saver is not None and cfg.RECOVERY_FREQ and (
                last_batch or (batch_idx + 1) % cfg.RECOVERY_FREQ == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates,
                                     metric=losses_m.avg)

        end = time.time()
    return OrderedDict([('loss', losses_m.avg)])


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup codebase default cfg.
    cfg = get_cfg()

    # Load config from cfg_file (load the configs that vary accross datasets).
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    # Load config from command line, overwrite config from opts (for the convenience of experiemnts).
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of XNN")

    # CUSTOMIZED
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int)
    parser.add_argument(
        '--gpus',
        dest="gpus",
        # nargs='+',
        type=int,
        default=0)
    parser.add_argument("--exp",
                        dest="EXPERIMENTAL_FEATURES",
                        help="customized experimental features",
                        default="",
                        type=str)
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See config/default_config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg = load_config(args)

    if cfg.DIST and cfg.NUM_GPUS > 1:
        # launch_job
        torch.multiprocessing.spawn(
            utils.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                main,
                cfg.INIT_METHOD,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                'nccl',
                cfg,
            ),
        )
    else:
        cfg.DIST = False
        main(cfg)