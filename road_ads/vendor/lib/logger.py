#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import logging

import torch.distributed as dist


class TensorboardXFilter(logging.Filter):
    """过滤 tensorboardX x2num.py 中的 NaN/Inf 警告"""
    def filter(self, record):
        # 过滤来自 x2num.py 的 NaN/Inf 警告
        if 'x2num' in record.filename and 'NaN or Inf' in record.getMessage():
            return False
        return True


def setup_logger(name, logpth, filter_tensorboard_nan=True):
    logfile = '{}-{}.log'.format(name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and dist.get_rank() != 0:
        log_level = logging.WARNING
    try:
        logging.basicConfig(level=log_level, format=FORMAT, filename=logfile, force=True)
    except Exception:
        for hl in logging.root.handlers: logging.root.removeHandler(hl)
        logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())
    
    # 过滤 tensorboardX 的 NaN 警告
    if filter_tensorboard_nan:
        # 为 tensorboardX.x2num logger 添加过滤器
        tensorboard_logger = logging.getLogger('tensorboardX.x2num')
        tensorboard_logger.addFilter(TensorboardXFilter())
        # 也为根 logger 添加，以防万一
        logging.root.addFilter(TensorboardXFilter())


def print_log_msg(it, max_iter, lr, time_meter, loss_meters, memory_meters):
    """
    打印日志信息，支持动态添加多个 loss。
    
    参数:
    - it: 当前迭代数
    - max_iter: 最大迭代数
    - lr: 当前学习率
    - time_meter: 时间计量器
    - loss_meters: 字典, key 为 loss 名称, value 为对应的 loss_meter
    """
    logger = logging.getLogger()
    t_intv, eta = time_meter.get()  # 获取时间间隔和预计时间
    # logger.info('eta: {}'.format(eta))
    loss_averages = {name: meter.get()[0] for name, meter in loss_meters.items()}  # 获取所有 loss 的平均值

    # 基础日志信息
    msg_parts = [
        f"iter: {it + 1}/{max_iter}",
        f"lr: {lr:.6f}",
        f"eta: {eta}",
        f"time: {t_intv:.2f}"
    ]

    # 动态添加 loss 信息
    msg_parts.extend([f"{name}: {avg:.4f}" for name, avg in loss_averages.items()])
    
    # 动态添加内存信息
    msg_parts.extend([f"{name}: {meter:.2f} GB" for name, meter in memory_meters.items()])

    # 合并消息
    msg = ', '.join(msg_parts)

    # 打印日志
    
    logger.info(msg)
