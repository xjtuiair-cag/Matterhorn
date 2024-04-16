# -*- coding: UTF-8 -*-
"""
数据的操作。
"""


import numpy as np
import torch
from typing import Iterable
from matterhorn_pytorch.__func__ import transpose
import matterhorn_pytorch.snn.functional as SF


def event_seq_to_spike_train(event_seq: torch.Tensor, shape: Iterable = None, original_shape: Iterable = None, count: bool = False) -> torch.Tensor:
    """
    将事件序列转为脉冲序列。
    Args:
        event_seq (torch.Tensor): 事件序列，形状为[N, A]
        shape (int*): 输出脉冲序列的形状
        original_shape (int*): 事件原本的画幅，若置空，则视为与脉冲序列形状一致
        count (bool): 是否输出事件计数，默认为False，即只输出脉冲（0或1）
    Return:
        spike_train (torch.Tensor): 脉冲序列，形状为[T, C, ...]
    """
    if not isinstance(event_seq, torch.Tensor):
        if isinstance(event_seq, np.ndarray):
            event_seq = event_seq.tolist()
        event_seq = torch.tensor(event_seq, dtype = torch.long)
    event_seq = event_seq.to(torch.float)
    if shape is None:
        shape = tuple([torch.max(event_seq[:, idx]) for idx in event_seq.shape[1]])
    if original_shape is None:
        original_shape = shape
    spike_train = torch.zeros(*shape, dtype = torch.float)
    for idx in range(len(shape)):
        if not event_seq.shape[0]:
            spike_train = spike_train.to(event_seq.device)
            return spike_train
        event_seq[:, idx] = torch.floor(event_seq[:, idx] * shape[idx] / original_shape[idx])
        event_filter = (event_seq[:, idx] >= 0) & (event_seq[:, idx] < shape[idx])
        event_seq = event_seq[event_filter]
    if not event_seq.shape[0]:
        spike_train = spike_train.to(event_seq.device)
        return spike_train
    event_seq, counts = torch.unique(event_seq, dim = 0, return_counts = True)
    event_seq = event_seq.permute(1, 0).to(torch.long)
    counts = counts.to(torch.float)
    spike_train[event_seq.tolist()] = (counts if count else 1.0)
    spike_train = spike_train.to(event_seq.device)
    return spike_train


def spike_train_to_event_seq(spike_train: torch.Tensor) -> torch.Tensor:
    """
    将脉冲序列转为事件序列。
    Args:
        spike_train (torch.Tensor): 脉冲序列，形状为[T, C, ...]
    Return:
        event_seq (torch.Tensor): 事件序列，形状为[N, A]
    """
    if not isinstance(spike_train, torch.Tensor):
        spike_train = torch.tensor(spike_train, dtype = torch.float)
    event_seq = spike_train.nonzero()
    event_seq = event_seq.to(spike_train.device)
    return event_seq


def spike_train_to_spike_times(spike_train: torch.Tensor, zero_fill: int = -1) -> torch.Tensor:
    """
    将脉冲序列转换为脉冲时间。
    Args:
        spike_train (torch.Tensor): 脉冲序列，形状为[T, ...]
        zero_fill (int | torch.inf): 无脉冲时的默认值，一般为-1，可以设为torch.inf
    Returns:
        spike_times (torch.Tensor): 时间序列，形状为[...]
    """
    spike_times = torch.where(torch.sum(spike_train, dim = 0).nonzero(), torch.argmax(spike_train, dim = 0), torch.full_like(spike_train[0], zero_fill))
    return spike_times


def spike_times_to_spike_train(spike_times: torch.Tensor, t_max: int, t_offset: int = 0) -> torch.Tensor:
    """
    将脉冲时间转换为脉冲序列。
    Args:
        spike_times (torch.Tensor): 时间序列，形状为[...]
        t_max (int): 最大时间步T
        t_offset (int): 时间步偏移量，从第几个时间步开始，一般为0
    Returns:
        spike_train (torch.Tensor): 脉冲序列，形状为[T, ...]
    """
    time_steps = t_max + t_offset
    spike_ts = torch.ones([time_steps] + list(spike_times.shape)) * (spike_times + t_offset)
    current_ts = transpose(transpose(torch.ones_like(spike_ts)) * torch.arange(time_steps)).to(spike_ts)
    spike_train = SF.ge(current_ts, spike_ts)
    return spike_train