# -*- coding: UTF-8 -*-
"""
数据的操作。
"""


import os
import shutil
import time
import random
import math
import urllib
import hashlib
import zipfile
import numpy as np
import torch
from torchvision.datasets.utils import download_url as _download_url, extract_archive as _extract_archive
from typing import Tuple as _Tuple, Iterable as _Iterable, Optional as _Optional
import matterhorn_pytorch.snn.functional as _SF
from rich import print


def get_md5(filename: str):
    md5_hash = hashlib.md5()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest().lower()


def download_file(src: str, dest: str, md5: _Optional[str] = None, retry: int = 5) -> None:
    md5 = md5.lower()
    if os.path.isfile(dest):
        if (md5 is not None) and get_md5(dest) != md5:
            os.remove(dest)
        else:
            print("[purple]File %s has already existed.[/purple]" % (dest,))
            return True
    if retry <= 0:
        print("[red]Failed to downloaded file %s from %s. Please manually download it.[/red]" % (dest, src))
        return False
    root, filename = os.path.split(dest)
    try:
        print("[blue]Trying to download %s from %s (retry=%d).[/blue]" % (dest, src, retry))
        _download_url(src, root = root, filename = filename, md5 = md5)
        print("[green]Successfully downloaded %s.[/green]" % (dest,))
        return True
    except urllib.error.URLError as e:
        print("[yellow]Error occured in downloading file %s from %s: %s.[/yellow]" % (dest, src, e))
        time.sleep(1.5 + random.random() * 1.0) # 防ban
        return download_file(src, dest, md5, retry - 1)
    except Exception as e:
        print("[yellow]Error occured in downloading file %s from %s: %s.[/yellow]" % (dest, src, e))
        return False


def check_zip_file(src: str, dest: str) -> _Tuple[bool, _Iterable[str]]:
    if not os.path.exists(dest):
        return False, []
    try:
        success = True
        res = []
        with zipfile.ZipFile(src, "r") as zip_ref:
            filelist = zip_ref.namelist()
            for filename in filelist:
                pathname = os.path.join(dest, filename)
                file_exists = os.path.exists(pathname)
                success = success and file_exists
                if file_exists:
                    res.append(pathname)
        return success, res
    except zipfile.BadZipFile as e:
        print("[yellow]Error occured in zip file %s: %s.[/yellow]" % (src, e))
        return False, []


def unzip_file(src: str, dest: str) -> str:
    success, existed_files = check_zip_file(src, dest)
    if success:
        print("[blue]File %s is already unzipped.[/blue]" % (src,))
        return True
    else:
        for filename in existed_files:
            if os.path.isfile(filename):
                os.remove(filename)
            if os.path.isdir(filename):
                shutil.rmtree(filename)
    try:
        print("[blue]Trying to unzip file %s ...[/blue]" % (src,))
        _extract_archive(src, dest)
        print("[green]Successfully unzip file %s.[/green]" % (src,))
        return True
    except zipfile.BadZipFile as e:
        print("[yellow]Error occured in unzipping file %s: %s.[/yellow]" % (src, e))
        return False


def event_seq_to_spike_train(event_seq: torch.Tensor, shape: _Iterable[int] = None, original_shape: _Iterable[int] = None, count: bool = False, dtype: torch.dtype = torch.float, device: torch.device = None) -> torch.Tensor:
    """
    将事件序列转为脉冲序列。
    Args:
        event_seq (torch.Tensor): 事件序列，形状为[N, A]
        shape (int*): 输出脉冲序列的形状
        original_shape (int*): 事件原本的画幅，若置空，则视为与脉冲序列形状一致
        count (bool): 是否输出事件计数，默认为False，即只输出脉冲（0或1）
        dtype (torch.dtype): 输出脉冲序列的数据类型
        device (torch.device): 输出脉冲序列的设备
    Return:
        spike_train (torch.Tensor): 脉冲序列，形状为[T, C, ...]
    """
    assert isinstance(event_seq, torch.Tensor), "The event sequence must be a tensor."
    if shape is None:
        shape = tuple([torch.max(event_seq[:, idx]) for idx in event_seq.shape[1]])
    if original_shape is None:
        original_shape = shape
    if device is None:
        device = event_seq.device
    assert event_seq.ndim == 2, "Need 2D input for the event sequence. %dD found." % (event_seq.ndim,)
    seq_len, ndim = event_seq.shape
    assert len(shape) == ndim, "The dimensions(columns) of event sequence aren't match. %d needed and %d received." % (len(shape), ndim)
    spike_train = torch.zeros(shape, dtype = dtype, device = device)
    if not seq_len:
        return spike_train
    indices = torch.zeros(seq_len)
    for dim in range(ndim):
        dim_indices = torch.clamp(torch.floor(event_seq[:, dim] * shape[dim] / original_shape[dim]), 0, shape[dim] - 1)
        indices = (indices + dim_indices) * (shape[dim + 1] if dim < ndim - 1 else 1)
    indices = indices.long()
    if count:
        spike_train.view(-1).scatter_add_(0, indices, torch.ones_like(indices, dtype = dtype))
    else:
        spike_train.view(-1).scatter_(0, indices, torch.ones_like(indices, dtype = dtype))
    return spike_train


def spike_train_to_event_seq(spike_train: torch.Tensor) -> torch.Tensor:
    """
    将脉冲序列转为事件序列。
    Args:
        spike_train (torch.Tensor): 脉冲序列，形状为[T, C, ...]
    Return:
        event_seq (torch.Tensor): 事件序列，形状为[N, A]
    """
    assert isinstance(spike_train, torch.Tensor), "The spike train must be a tensor."
    event_seq = spike_train.nonzero()
    event_seq = event_seq.to(spike_train)
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
    spike_times = torch.where(torch.sum(spike_train, dim = 0).gt(0.0), torch.argmax(spike_train, dim = 0).to(spike_train), torch.full_like(spike_train[0], zero_fill))
    return spike_times


def spike_times_to_spike_train(spike_times: torch.Tensor, time_steps: int, t_offset: int = 0) -> torch.Tensor:
    """
    将脉冲时间转换为脉冲序列。
    Args:
        spike_times (torch.Tensor): 时间序列，形状为[...]
        time_steps (int): 最大时间步T
        t_offset (int): 时间步偏移量，从第几个时间步开始
    Returns:
        spike_train (torch.Tensor): 脉冲序列，形状为[T, ...]
    """
    spike_train = _SF.encode_temporal(spike_times, time_steps, t_offset)
    return spike_train