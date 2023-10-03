# -*- coding: UTF-8 -*-
"""
画图小工具，利用matplotlib库，将数据（事件、电位等）以图片的形式打印下来。
"""


import numpy as np
import torch
import re
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Iterable, Union, Optional


def transition_color(val: float, min_val: float = 0., max_val: float = 1., min_color: str = "#000000", max_color: str = "#ffffff"):
    """
    获取过渡色。
    @params:
        val: float 当前值
        min_val: float 最小值
        max_val: float 最大值
        min_color: str 当数据处于最小值时，点应当有的颜色
        max_color: str 当数据处于最大值时，点应当有的颜色
    @return:
        hex: str 该值所对应颜色的hex编码
    """
    min_color = "0x" + re.sub(r"[^0-9abcdefxABCDEFX]", "", min_color)
    max_color = "0x" + re.sub(r"[^0-9abcdefxABCDEFX]", "", max_color)
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


def event_plot_1d(data: torch.Tensor, indices_pos: np.ndarray, indices_neg: np.ndarray = None, color_pos: str = "#0000ff", color_neg: str = "#ff0000", title: str = None, index_0: str = "t", prec: int = 8) -> None:
    """
    单次一维事件数据打印。
    @params:
        indices_pos: np.ndarray （正）事件索引
        indices_neg: np.ndarray 负事件索引，如果没有极性传入None
        color_pos: str 正事件颜色（Hex）
        color_neg: str 负事件颜色（Hex）
        title: str 图片的标题
        index_0: str 维度的索引
        prec: int 事件的脉冲有多细
    """
    if not indices_pos.shape[0]:
        return
    indices_0 = np.arange(prec * (np.max(indices_pos) + 1)) / prec
    indices_1_pos = np.zeros_like(indices_0)
    indices_1_pos[indices_pos * prec] = 1
    plt.plot(indices_0, indices_1_pos, c = color_pos)
    if indices_neg is not None and indices_neg.shape[0]:
        indices_1_neg = np.zeros_like(indices_0)
        indices_1_neg[indices_neg * prec] = 1
        plt.plot(indices_0, indices_1_neg, c = color_neg)
    plt.xlabel(index_0)
    if title is not None:
        plt.title(title)


def event_plot_2d(data: torch.Tensor, indices_pos: np.ndarray, indices_neg: np.ndarray = None, color_pos: str = "#0000ff", color_neg: str = "#ff0000", title: str = None, index_0: str = "t", index_1: str = "x", shift: float = 0.1, color_depth: int = 8) -> None:
    """
    单次二维事件数据打印。
    @params:
        indices_pos: np.ndarray （正）事件索引
        indices_neg: np.ndarray 负事件索引，如果没有极性传入None
        color_pos: str 正事件颜色（Hex）
        color_neg: str 负事件颜色（Hex）
        title: str 图片的标题
        index_0: str 最外维度的索引
        index_1: str 最内维度的索引
        shift: float 给事件加上的偏移量，方便同时打印正负事件
        color_depth: int 如果是多值事件，需要分多少种颜色表示
    """
    if not indices_pos.shape[0]:
        return
    polarity_exists = (len(data.shape) > 2)
    pos_vals = (data[0] if polarity_exists else data)[indices_pos].numpy()
    min_val = np.min(pos_vals)
    max_val = np.max(pos_vals)
    color_val = ((pos_vals - min_val) / (max_val - min_val)) if max_val > min_val else 1.0
    for i in range(0, color_depth):
        points = (color_val > (i / color_depth)) & (color_val <= ((i + 1) / color_depth))
        if not np.any(points):
            continue
        color = transition_color(i + 1, 0, color_depth, "#7f7f7f", color_pos)
        plt.scatter(indices_pos[0, points] + shift, indices_pos[1, points] + shift, s = 1, c = color)
    if indices_neg is not None and indices_neg.shape[0]:
        neg_vals = (data[1] if polarity_exists else data)[indices_neg].numpy()
        min_val = np.min(neg_vals)
        max_val = np.max(neg_vals)
        color_val = ((neg_vals - min_val) / (max_val - min_val)) if max_val > min_val else 1.0
        for i in range(0, color_depth):
            points = (color_val > (i / color_depth)) & (color_val <= ((i + 1) / color_depth))
            if not np.any(points):
                continue
            color = transition_color(i + 1, 0, color_depth, "#7f7f7f", color_neg)
            plt.scatter(indices_neg[0, points] - shift, indices_neg[1, points] - shift, s = 1, c = color)
    plt.xlabel(index_0)
    plt.ylabel(index_1)
    plt.xlim(-0.5, data.shape[1 if polarity_exists else 0] - 0.5)
    plt.ylim(-0.5, data.shape[2 if polarity_exists else 1] - 0.5)
    if title is not None:
        plt.title(title)


def event_plot_yx(data: torch.Tensor, polarity: bool = True, show: bool = True, save: str = None, titles: Iterable[str] = None) -> None:
    """
    二维空间的事件数据打印。
    @params:
        data: torch.Tensor 需要打印的数据（可以是2、3、4维）
        polarity: bool 是否将第2个维度作为极性维度，如果为True，则传入的数据为[C(P), H, W]（3维）或[B, C(P), H, W]（4维）；如果为False，则传入的数据为[H, W]（2维）或[B, H, W]（3维）
        show: bool 是否展示图像，默认展示
        save: str 是否保存图像，若传入路径，则保存图像；否则不保存
        titles: List[str] 每张图都是什么标题
    """
    get_title = lambda b: titles[b] if titles is not None and b >= 0 and b < len(titles) else ("Event Group %d" % (b,))
    dim = len(data.shape)
    data_numpy = data.numpy()
    indices = np.array(data_numpy.nonzero())
    fig = plt.figure()
    if dim == 2: # [H, W]
        event_plot_2d(data.permute(1, 0), indices[[1, 0]], color_pos = "#00ff00", title = get_title(0), index_0 = "x", index_1 = "y")
    if dim == 3:
        if polarity: # [C(P), H, W]
            event_plot_2d(data.permute(0, 2, 1), indices[:, indices[0] == 1][[2, 1]], indices[:, indices[0] == 0][[2, 1]], title = get_title(0), index_0 = "x", index_1 = "y")
        else: # [B, H, W]
            batch_size = data_numpy.shape[0]
            rows = batch_size
            cols = 1
            if not rows % 2:
                rows = rows // 2
                cols = cols * 2
            for b in range(batch_size):
                plt.subplot(rows, cols, b + 1)
                event_plot_2d(data[b].permute(1, 0), indices[:, indices[0] == b][[2, 1]], color_pos = "#00ff00", title = get_title(b), index_0 = "x", index_1 = "y")
    if dim == 4: # [B, C(P), H, W]
        batch_size = data_numpy.shape[0]
        rows = batch_size
        cols = 1
        if not rows % 2:
            rows = rows // 2
            cols = cols * 2
        for b in range(batch_size):
            plt.subplot(rows, cols, b + 1)
            event_plot_2d(data[b].permute(0, 2, 1), indices[:, (indices[0] == b) & (indices[1] == 1)][[3, 2]], indices[:, (indices[0] == b) & (indices[1] == 0)][[3, 2]], title = get_title(b), index_0 = "x", index_1 = "y")
    if show:
        fig.canvas.set_window_title("Event Plotter Result")
        plt.show()
    if save is not None:
        plt.savefig(save)
        plt.close()


def event_plot_tx(data: torch.Tensor, polarity: bool = True, show: bool = True, save: str = None, titles: Iterable[str] = None) -> None:
    """
    一维空间+一维时间的事件数据打印。
    @params:
        data: torch.Tensor 需要打印的数据（可以是2、3、4维）
        polarity: bool 是否将第2个维度作为极性维度，如果为True，则传入的数据为[T, C(P), L]（3维）或[B, T, C(P), L]（4维）；如果为False，则传入的数据为[T, L]（2维）或[B, T, L]（3维）
        show: bool 是否展示图像，默认展示
        save: str 是否保存图像，若传入路径，则保存图像；否则不保存
        titles: List[str] 每张图都是什么标题
    """
    get_title = lambda b: titles[b] if titles is not None and b >= 0 and b < len(titles) else ("Event Group %d" % (b,))
    dim = len(data.shape)
    data_numpy = data.numpy()
    indices = np.array(data_numpy.nonzero())
    fig = plt.figure()
    if dim == 2: # [T, L]
        event_plot_2d(data, indices, color_pos = "#00ff00", title = get_title(0))
    if dim == 3:
        if polarity: # [T, C(P), L]
            event_plot_2d(data.permute(1, 0, 2), indices[:, indices[1] == 1][[0, 2]], indices[:, indices[1] == 0][[0, 2]], title = get_title(0))
        else: # [B, T, L]
            batch_size = data_numpy.shape[0]
            rows = batch_size
            cols = 1
            if not rows % 2:
                rows = rows // 2
                cols = cols * 2
            for b in range(batch_size):
                plt.subplot(rows, cols, b + 1)
                event_plot_2d(data[b], indices[:, indices[0] == b][[1, 2]], color_pos = "#00ff00", title = get_title(b))
    if dim == 4: # [B, T, C(P), L]
        batch_size = data_numpy.shape[0]
        rows = batch_size
        cols = 1
        if not rows % 2:
            rows = rows // 2
            cols = cols * 2
        for b in range(batch_size):
            plt.subplot(rows, cols, b + 1)
            event_plot_2d(data[b].permute(1, 0, 2), indices[:, (indices[0] == b) & (indices[2] == 1)][[1, 3]], indices[:, (indices[0] == b) & (indices[2] == 0)][[1, 3]], title = get_title(b))
    if show:
        fig.canvas.set_window_title("Event Plotter Result")
        plt.show()
    if save is not None:
        plt.savefig(save)
        plt.close()


def event_plot_3d(ax: Axes3D, data: torch.Tensor, indices_pos: np.ndarray, indices_neg: np.ndarray = None, color_pos: str = "#0000ff", color_neg: str = "#ff0000", title: str = None, index_0: str = "t", index_1: str = "y", index_2: str = "x", shift: float = 0.1, color_depth: int = 8) -> None:
    """
    单次三维事件数据打印。
    @params:
        ax: Axes3D 画布
        indices_pos: np.ndarray （正）事件索引
        indices_neg: np.ndarray 负事件索引，如果没有极性传入None
        color_pos: str 正事件颜色（Hex）
        color_neg: str 负事件颜色（Hex）
        title: str 图片的标题
        index_0: str 最外维度的索引
        index_1: str 中间维度的索引
        index_2: str 最内维度的索引
        shift: float 给事件加上的偏移量，方便同时打印正负事件
        color_depth: int 如果是多值事件，需要分多少种颜色表示
    """
    if not indices_pos.shape[0]:
        return
    polarity_exists = (len(data.shape) > 3)
    pos_vals = (data[0] if polarity_exists else data)[indices_pos].numpy()
    min_val = np.min(pos_vals)
    max_val = np.max(pos_vals)
    color_val = ((pos_vals - min_val) / (max_val - min_val)) if max_val > min_val else 1.0
    for i in range(0, color_depth):
        points = (color_val > (i / color_depth)) & (color_val <= ((i + 1) / color_depth))
        if not np.any(points):
            continue
        color = transition_color(i + 1, 0, color_depth, "#7f7f7f", color_pos)
        ax.scatter(indices_pos[2, points] + shift, indices_pos[0, points] + shift, indices_pos[1, points] + shift, s = 1, c = color)
    if indices_neg is not None and indices_neg.shape[0]:
        neg_vals = (data[1] if polarity_exists else data)[indices_neg].numpy()
        min_val = np.min(neg_vals)
        max_val = np.max(neg_vals)
        color_val = ((neg_vals - min_val) / (max_val - min_val)) if max_val > min_val else 1.0
        for i in range(0, color_depth):
            points = (color_val > (i / color_depth)) & (color_val <= ((i + 1) / color_depth))
            if not np.any(points):
                continue
            color = transition_color(i + 1, 0, color_depth, "#7f7f7f", color_neg)
            ax.scatter(indices_neg[2, points] - shift, indices_neg[0, points] - shift, indices_neg[1, points] - shift, s = 1, c = color)
    ax.set_xlabel(index_2)
    ax.set_ylabel(index_0)
    ax.set_zlabel(index_1)
    ax.set_xlim(-0.5, data.shape[3 if polarity_exists else 2] - 0.5)
    ax.set_ylim(-0.5, data.shape[1 if polarity_exists else 0] - 0.5)
    ax.set_zlim3d(-0.5, data.shape[2 if polarity_exists else 1] - 0.5)
    if title is not None:
        ax.set_title(title)


def event_plot_tyx(data: torch.Tensor, polarity: bool = True, show: bool = True, save: str = None, titles: Iterable[str] = None) -> None:
    """
    二维空间+一维时间的事件数据打印。
    @params:
        data: torch.Tensor 需要打印的数据（可以是3、4、5维）
        polarity: bool 是否将第2个维度作为极性维度，如果为True，则传入的数据为[T, C(P), H, W]（4维）或[B, T, C(P), H, W]（5维）；如果为False，则传入的数据为[T, H, W]（3维）或[B, T, H, W]（4维）
        show: bool 是否展示图像，默认展示
        save: str 是否保存图像，若传入路径，则保存图像；否则不保存
        titles: List[str] 每张图都是什么标题
    """
    get_title = lambda b: titles[b] if titles is not None and b >= 0 and b < len(titles) else ("Event Group %d" % (b,))
    dim = len(data.shape)
    data_numpy = data.numpy()
    indices = np.array(data_numpy.nonzero())
    fig = plt.figure()
    if dim == 3: # [T, H, W]
        ax = fig.add_subplot(projection='3d')
        event_plot_3d(ax, data, indices, color_pos = "#00ff00", title = get_title(0))
    if dim == 4:
        if polarity: # [T, C(P), H, W]
            ax = fig.add_subplot(projection='3d')
            event_plot_3d(ax, data.permute(1, 0, 2, 3), indices[:, indices[1] == 1][[0, 2, 3]], indices[:, indices[1] == 0][[0, 2, 3]], title = get_title(0))
        else: # [B, T, H, W]
            batch_size = data_numpy.shape[0]
            rows = batch_size
            cols = 1
            if not rows % 3:
                rows = rows // 3
                cols = cols * 3
            elif not rows % 2:
                rows = rows // 2
                cols = cols * 2
            for b in range(batch_size):
                ax = fig.add_subplot(rows, cols, b + 1, projection='3d')
                event_plot_3d(ax, data[b], indices[:, indices[0] == b][[1, 2, 3]], color_pos = "#00ff00", title = get_title(b))
    if dim == 5: # [B, T, C(P), H, W]
        batch_size = data_numpy.shape[0]
        rows = batch_size
        cols = 1
        if not rows % 3:
            rows = rows // 3
            cols = cols * 3
        elif not rows % 2:
            rows = rows // 2
            cols = cols * 2
        for b in range(batch_size):
            ax = fig.add_subplot(rows, cols, b + 1, projection='3d')
            event_plot_3d(ax, data[b].permute(1, 0, 2, 3), indices[:, (indices[0] == b) & (indices[2] == 1)][[1, 3, 4]], indices[:, (indices[0] == b) & (indices[2] == 0)][[1, 3, 4]], title = get_title(b))
    if show:
        fig.canvas.set_window_title("Event Plotter Result")
        plt.show()
    if save is not None:
        plt.savefig(save)
        plt.close()