# -*- coding: UTF-8 -*-
"""
画图小工具，利用matplotlib库，将数据（事件、电位等）以图片的形式打印下来。
"""


import re
import numpy as np
import torch
import matterhorn_pytorch.data.functional as _DF
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from typing import Any, Tuple, Iterable, Mapping, Union, Optional


def transition_color(val: float, min_val: float = 0., max_val: float = 1., min_color: str = "#000000", max_color: str = "#ffffff"):
    """
    获取过渡色。
    Args:
        val (float): 当前值
        min_val (float): 最小值
        max_val (float): 最大值
        min_color (str): 当数据处于最小值时，点应当有的颜色
        max_color (str): 当数据处于最大值时，点应当有的颜色
    Returns:
        hex (str): 该值所对应颜色的hex编码
    """
    min_color = "0x" + re.sub(r"[^0-9abcdefABCDEF]", "", min_color)
    max_color = "0x" + re.sub(r"[^0-9abcdefABCDEF]", "", max_color)
    source = int(eval(min_color))
    dest = int(eval(max_color))
    source_r, source_g, source_b = (source >> 16) & 0xff, (source >> 8) & 0xff, source & 0xff
    dest_r, dest_g, dest_b = (dest >> 16) & 0xff, (dest >> 8) & 0xff, dest & 0xff
    delta_r, delta_g, delta_b = dest_r - source_r, dest_g - source_g, dest_b - source_b
    ratio = (val - min_val) / (max_val - min_val)
    target_r, target_g, target_b = int(max(0, min(255, source_r + delta_r * ratio))), int(max(0, min(255, source_g + delta_g * ratio))), int(max(0, min(255, source_b + delta_b * ratio)))
    res = hex((target_r << 16) + (target_g << 8) + target_b).replace("0x", "#")
    if len(res) < 7:
        res = "#" + ("0" * (7 - len(res))) + res[1:]
    return res


def init_figure(ndim: int, **kwargs) -> Tuple[Figure, Union[Axes, Axes3D]]:
    fig = plt.figure(**kwargs)
    if ndim >= 3:
        ax = fig.add_subplot(projection = "3d")
    else:
        ax = plt.subplot()
    return fig, ax


def _check_and_merge_kwargs(kwargs: Mapping, ndim: int, default_kwargs: Mapping = dict()) -> Mapping:
    for k in kwargs:
        if k in ("color_pos", "color_neg"):
            default_kwargs[k] = kwargs[k]
        if k in tuple("index_%d" % (d,) for d in range(ndim)):
            default_kwargs[k] = kwargs[k]
        if k in ("prec",) and ndim == 1:
            default_kwargs[k] = kwargs[k]
        if k in ("shift",) and ndim in (2, 3):
            default_kwargs[k] = kwargs[k]
    return default_kwargs


def event_plot_1d(ax: Axes, indices_pos: np.ndarray, indices_neg: np.ndarray = None, color_pos: str = "#0000ff", color_neg: str = "#ff0000", index_0: str = "t", prec: int = 8) -> None:
    """
    单次一维事件数据打印。
    Args:
        indices_pos (np.ndarray): （正）事件索引
        indices_neg (np.ndarray): 负事件索引，如果没有极性传入None
        color_pos (str): 正事件颜色（Hex）
        color_neg (str): 负事件颜色（Hex）
        index_0 (str): 维度的索引
        prec (int): 事件的脉冲有多细
    """
    def draw(ax: Axes, indices: np.ndarray, color: str) -> None:
        indices_0 = np.arange(prec * (np.max(indices) + 1)) / prec
        indices_1 = np.zeros_like(indices_0)
        indices_1[indices * prec] = 1
        ax.plot(indices_0, indices_1, c = color)

    if indices_pos.shape[0]:
        draw(ax, indices_pos, color_pos)
    if indices_neg is not None and indices_neg.shape[0]:
        draw(ax, indices_neg, color_neg)
    ax.set_xlabel(index_0)


def event_plot_2d(ax: Axes, indices_pos: np.ndarray, indices_neg: np.ndarray = None, values_pos: np.ndarray = None, values_neg: np.ndarray = None, color_pos: str = "#0000ff", color_neg: str = "#ff0000", index_0: str = "t", index_1: str = "x", shape: Tuple = None, shift: float = 0.1) -> None:
    """
    单次二维事件数据打印。
    Args:
        indices_pos (np.ndarray): （正）事件索引
        indices_neg (np.ndarray): 负事件索引，如果没有极性传入None
        values_pos (np.ndarray): 正事件值，默认全为1
        values_pos (np.ndarray): 负事件值，默认全为1
        color_pos (str): 正事件颜色（Hex）
        color_neg (str): 负事件颜色（Hex）
        index_0 (str): 最外维度的索引
        index_1 (str): 最内维度的索引
        shape (Iterable): 画幅的形状
        shift (float): 给事件加上的偏移量，方便同时打印正负事件
    """
    def draw(ax: Axes, indices: np.ndarray, values: np.ndarray, color: str, shift: float) -> None:
        if values is None:
            values = np.ones_like(indices)
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val > min_val:
            color_val = ((values - min_val) / (max_val - min_val))
            colors = [transition_color(0.5 + v * 0.5, min_color = "#7f7f7f", max_color = color) for v in color_val]
        else:
            color_val = 1.0
            colors = color
        ax.scatter(indices[:, 0] + shift, indices[:, 1] + shift, s = 1, c = colors)

    if indices_pos.shape[0]:
        draw(ax, indices_pos, values_pos, color_pos, shift)
    if indices_neg is not None and indices_neg.shape[0]:
        draw(ax, indices_neg, values_neg, color_neg, -shift)
    ax.set_xlabel(index_0)
    ax.set_ylabel(index_1)
    if shape is not None and len(shape) == 2:
        ax.set_xlim(-0.5, shape[0] - 0.5)
        ax.set_ylim(-0.5, shape[1] - 0.5)


def event_plot_3d(ax: Axes3D, indices_pos: np.ndarray, indices_neg: np.ndarray = None, values_pos: np.ndarray = None, values_neg: np.ndarray = None, color_pos: str = "#0000ff", color_neg: str = "#ff0000", index_0: str = "t", index_1: str = "y", index_2: str = "x", shape: Tuple = None, shift: float = 0.1) -> None:
    """
    单次三维事件数据打印。
    Args:
        ax (Axes3D): 画布
        indices_pos (np.ndarray): （正）事件索引
        indices_neg (np.ndarray): 负事件索引，如果没有极性传入None
        values_pos (np.ndarray): 正事件值，默认全为1
        values_pos (np.ndarray): 负事件值，默认全为1
        color_pos (str): 正事件颜色（Hex）
        color_neg (str): 负事件颜色（Hex）
        index_0 (str): 最外维度的索引
        index_1 (str): 中间维度的索引
        index_2 (str): 最内维度的索引
        shape (Iterable): 画幅的形状
        shift (float): 给事件加上的偏移量，方便同时打印正负事件
    """
    def draw(ax: Axes3D, indices: np.ndarray, values: np.ndarray, color: str, shift: float) -> None:
        if values is None:
            values = np.ones_like(indices)
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val > min_val:
            color_val = ((values - min_val) / (max_val - min_val))
            colors = [transition_color(0.5 + v * 0.5, min_color = "#7f7f7f", max_color = color) for v in color_val]
        else:
            color_val = 1.0
            colors = color
        ax.scatter(indices[:, 0] + shift, indices[:, 2] + shift, indices[:, 1] + shift, s = 1, c = colors)
    
    if indices_pos.shape[0]:
        draw(ax, indices_pos, values_pos, color_pos, shift)
    if indices_neg is not None and indices_neg.shape[0]:
        draw(ax, indices_neg, values_neg, color_neg, -shift)
    ax.set_xlabel(index_0)
    ax.set_ylabel(index_2)
    ax.set_zlabel(index_1)
    if shape is not None and len(shape) == 3:
        ax.set_xlim3d(-0.5, shape[0] - 0.5)
        ax.set_ylim3d(-0.5, shape[2] - 0.5)
        ax.set_zlim3d(-0.5, shape[1] - 0.5)


def spike_train_plot_yx(data: torch.Tensor, ax: Axes = None, **kwargs) -> Axes:
    """
    二维空间的脉冲序列数据打印。
    Args:
        data (torch.Tensor):需要打印的数据（可以是2、3维）
        ax (Axes): 图坐标轴
    Returns:
        ax (Axes): 图坐标轴
    """
    if ax is None:
        fig, ax = init_figure(ndim = 2)

    dim = data.ndim
    indices = _DF.spike_train_to_event_seq(data).long()
    values = data[tuple(indices[:, d] for d in range(indices.shape[1]))]

    default = dict(
        index_0 = "x",
        index_1 = "y"
    )
    if dim == 2:
        default = _check_and_merge_kwargs(dict(
            color_pos = "#ff00ff"
        ), 2, default)
        event_plot_2d(
            ax = ax,
            indices_pos = indices[:, [1, 0]].numpy(),
            values_pos = values.numpy(),
            **_check_and_merge_kwargs(kwargs, 2, default)
        )
    elif dim == 3:
        pos_mask = (indices[:, 0] >= (data.shape[0] // 2))
        neg_mask = (indices[:, 0] < (data.shape[0] // 2))
        event_plot_2d(
            ax = ax,
            indices_pos = indices[pos_mask][:, [2, 1]].numpy(),
            indices_neg = indices[neg_mask][:, [2, 1]].numpy(),
            values_pos = values[pos_mask].numpy(),
            values_neg = values[neg_mask].numpy(),
            **_check_and_merge_kwargs(kwargs, 2, default)
        )
    else:
        raise ValueError("Invalid dimension: %d." % (dim,))
    return ax


def event_seq_plot_yx(data: torch.Tensor, shape: Iterable, ax: Axes = None, **kwargs) -> Axes:
    """
    二维空间的事件序列数据打印。
    Args:
        data (torch.Tensor): 需要打印的数据（可以是2、3维）
        shape (Iterable): 图的形状（长度为data列数的元组），若不指定（或为0）则为数据的长度
        ax (Axes): 图坐标轴
    Returns:
        ax (Axes): 图坐标轴
    """
    return spike_train_plot_yx(
        data = _DF.event_seq_to_spike_train(
            event_seq = data,
            shape = shape,
            dtype = data.dtype,
            **_check_and_merge_kwargs(kwargs, 2)
        ),
        ax = ax
    )


def event_plot_yx(data: torch.Tensor, shape: Iterable = None, is_seq: bool = False, ax: Axes = None, **kwargs) -> Axes:
    """
    二维空间的事件数据打印。
    Args:
        data (torch.Tensor): 需要打印的数据
        shape (Iterable): 图的形状（打印序列用）
        is_seq (bool): 是否为序列，True表明事件为形状为[n, 2]或[n, 3]的序列，否则为形状为[T, C(P), H, W]的张量
        ax (Axes): 图坐标轴
    Returns:
        ax (Axes): 图坐标轴
    """
    if ax is None:
        fig, ax = init_figure(ndim = 2)
    if is_seq:
        return event_seq_plot_yx(
            data = data,
            shape = shape,
            ax = ax,
            **_check_and_merge_kwargs(kwargs, 2)
        )
    else:
        return spike_train_plot_yx(
            data = data,
            ax = ax,
            **_check_and_merge_kwargs(kwargs, 2)
        )


def spike_train_plot_tx(data: torch.Tensor, ax: Axes = None, **kwargs) -> Axes:
    """
    一维空间+一维时间的脉冲序列数据打印。
    Args:
        data (torch.Tensor):需要打印的数据（可以是2、3维）
        ax (Axes): 图坐标轴
    Returns:
        ax (Axes): 图坐标轴
    """
    if ax is None:
        fig, ax = init_figure(ndim = 2)

    dim = data.ndim
    indices = _DF.spike_train_to_event_seq(data).long()
    values = data[tuple(indices[:, d] for d in range(indices.shape[1]))]

    default = dict()
    if dim == 2:
        default = _check_and_merge_kwargs(dict(
            color_pos = "#ff00ff"
        ), 2, default)
        event_plot_2d(
            ax = ax,
            indices_pos = indices[:, [0, 1]].numpy(),
            values_pos = values.numpy(),
            **_check_and_merge_kwargs(kwargs, 2, default)
        )
    elif dim == 3:
        pos_mask = (indices[:, 1] >= (data.shape[1] // 2))
        neg_mask = (indices[:, 1] < (data.shape[1] // 2))
        event_plot_2d(
            ax = ax,
            indices_pos = indices[pos_mask][:, [0, 2]].numpy(),
            indices_neg = indices[neg_mask][:, [0, 2]].numpy(),
            values_pos = values[pos_mask].numpy(),
            values_neg = values[neg_mask].numpy(),
            **_check_and_merge_kwargs(kwargs, 2, default)
        )
    else:
        raise ValueError("Invalid dimension: %d." % (dim,))
    return ax


def event_seq_plot_tx(data: torch.Tensor, shape: Iterable, ax: Axes = None, **kwargs) -> Axes:
    """
    一维空间+一维时间的事件序列数据打印。
    Args:
        data (torch.Tensor): 需要打印的数据（可以是2、3维）
        shape (Iterable): 图的形状（长度为data列数的元组），若不指定（或为0）则为数据的长度
        ax (Axes): 图坐标轴
    Returns:
        ax (Axes): 图坐标轴
    """
    return spike_train_plot_tx(
        data = _DF.event_seq_to_spike_train(
            event_seq = data,
            shape = shape,
            dtype = data.dtype,
            **_check_and_merge_kwargs(kwargs, 2)
        ),
        ax = ax
    )


def event_plot_tx(data: torch.Tensor, shape: Iterable = None, is_seq: bool = False, ax: Axes = None, **kwargs) -> Axes:
    """
    一维空间+一维时间的事件数据打印。
    Args:
        data (torch.Tensor): 需要打印的数据
        shape (Iterable): 图的形状（打印序列用）
        is_seq (bool): 是否为序列，True表明事件为形状为[n, 2]或[n, 3]的序列，否则为形状为[T, C(P), H, W]的张量
        ax (Axes): 图坐标轴
    Returns:
        ax (Axes): 图坐标轴
    """
    if ax is None:
        fig, ax = init_figure(ndim = 2)
    if is_seq:
        return event_seq_plot_tx(
            data = data,
            shape = shape,
            ax = ax,
            **_check_and_merge_kwargs(kwargs, 2)
        )
    else:
        return spike_train_plot_tx(
            data = data,
            ax = ax,
            **_check_and_merge_kwargs(kwargs, 2)
        )


def spike_train_plot_tyx(data: torch.Tensor, ax: Axes = None, **kwargs) -> Axes:
    """
    二维空间+一维时间的脉冲序列数据打印。
    Args:
        data (torch.Tensor):需要打印的数据（可以是3、4维）
        ax (Axes): 图坐标轴
    Returns:
        ax (Axes): 图坐标轴
    """
    if ax is None:
        fig, ax = init_figure(ndim = 3)

    dim = data.ndim
    indices = _DF.spike_train_to_event_seq(data).long()
    values = data[tuple(indices[:, d] for d in range(indices.shape[1]))]

    default = dict()
    if dim == 3:
        default = _check_and_merge_kwargs(dict(
            color_pos = "#ff00ff"
        ), 3, default)
        event_plot_3d(
            ax = ax,
            indices_pos = indices[:, [0, 1, 2]].numpy(),
            values_pos = values.numpy(),
            **_check_and_merge_kwargs(kwargs, 3, default)
        )
    elif dim == 4:
        pos_mask = (indices[:, 1] >= (data.shape[1] // 2))
        neg_mask = (indices[:, 1] < (data.shape[1] // 2))
        event_plot_3d(
            ax = ax,
            indices_pos = indices[pos_mask][:, [0, 2, 3]].numpy(),
            indices_neg = indices[neg_mask][:, [0, 2, 3]].numpy(),
            values_pos = values[pos_mask].numpy(),
            values_neg = values[neg_mask].numpy(),
            **_check_and_merge_kwargs(kwargs, 3)
        )
    else:
        raise ValueError("Invalid dimension: %d." % (dim,))
    return ax


def event_seq_plot_tyx(data: torch.Tensor, shape: Iterable, ax: Axes = None, **kwargs) -> Axes:
    """
    二维空间+一维时间的事件序列数据打印。
    Args:
        data (torch.Tensor): 需要打印的数据（可以是3、4维）
        shape (Iterable): 图的形状（长度为data列数的元组），若不指定（或为0）则为数据的长度
        ax (Axes): 图坐标轴
    Returns:
        ax (Axes): 图坐标轴
    """
    return spike_train_plot_tyx(
        data = _DF.event_seq_to_spike_train(
            event_seq = data,
            shape = shape,
            dtype = data.dtype
        ),
        ax = ax,
        **_check_and_merge_kwargs(kwargs, 3)
    )


def event_plot_tyx(data: torch.Tensor, shape: Iterable = None, is_seq: bool = False, ax: Axes = None, **kwargs) -> Axes:
    """
    二维空间+一维时间的事件数据打印。
    Args:
        data (torch.Tensor): 需要打印的数据
        shape (Iterable): 图的形状（打印序列用）
        is_seq (bool): 是否为序列，True表明事件为形状为[n, 3]或[n, 4]的序列，否则为形状为[T, C(P), H, W]的张量
        ax (Axes): 图坐标轴
    Returns:
        ax (Axes): 图坐标轴
    """
    if ax is None:
        fig, ax = init_figure(ndim = 3)
    if is_seq:
        return event_seq_plot_tyx(
            data = data,
            shape = shape,
            ax = ax,
            **_check_and_merge_kwargs(kwargs, 3)
        )
    else:
        return spike_train_plot_tyx(
            data = data,
            ax = ax,
            **_check_and_merge_kwargs(kwargs, 3)
        )