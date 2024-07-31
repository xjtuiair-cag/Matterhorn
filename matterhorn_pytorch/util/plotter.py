# -*- coding: UTF-8 -*-
"""
画图小工具，利用matplotlib库，将数据（事件、电位等）以图片的形式打印下来。
"""


import numpy as np
import torch
import os
import re
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties, findSystemFonts
from typing import Tuple, Iterable, Union, Optional


plt.rcParams['font.family'] = ['Helvetica Neue', 'Arial']
plt.rcParams['mathtext.fontset'] = 'stix'


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


def event_plot_1d(indices_pos: np.ndarray, indices_neg: np.ndarray = None, color_pos: str = "#0000ff", color_neg: str = "#ff0000", title: str = None, index_0: str = "t", prec: int = 8) -> None:
    """
    单次一维事件数据打印。
    Args:
        indices_pos (np.ndarray): （正）事件索引
        indices_neg (np.ndarray): 负事件索引，如果没有极性传入None
        color_pos (str): 正事件颜色（Hex）
        color_neg (str): 负事件颜色（Hex）
        title (str): 图片的标题
        index_0 (str): 维度的索引
        prec (int): 事件的脉冲有多细
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


def event_plot_2d(indices_pos: np.ndarray, indices_neg: np.ndarray = None, values_pos: np.ndarray = None, values_neg: np.ndarray = None, color_pos: str = "#0000ff", color_neg: str = "#ff0000", title: str = None, index_0: str = "t", index_1: str = "x", shape: Tuple = None, shift: float = 0.1) -> None:
    """
    单次二维事件数据打印。
    Args:
        indices_pos (np.ndarray): （正）事件索引
        indices_neg (np.ndarray): 负事件索引，如果没有极性传入None
        values_pos (np.ndarray): 正事件值，默认全为1
        values_pos (np.ndarray): 负事件值，默认全为1
        color_pos (str): 正事件颜色（Hex）
        color_neg (str): 负事件颜色（Hex）
        title (str): 图片的标题
        index_0 (str): 最外维度的索引
        index_1 (str): 最内维度的索引
        shape (Iterable): 画幅的形状
        shift (float): 给事件加上的偏移量，方便同时打印正负事件
    """
    def draw(indices: np.ndarray, values: np.ndarray, color: str, shift: float) -> None:
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
        plt.scatter(indices[:, 0] + shift, indices[:, 1] + shift, s = 1, c = colors)

    if indices_pos.shape[0]:
        draw(indices_pos, values_pos, color_pos, shift)
    if indices_neg is not None and indices_neg.shape[0]:
        draw(indices_neg, values_neg, color_neg, -shift)
    plt.xlabel(index_0)
    plt.ylabel(index_1)
    if shape is not None and len(shape) == 2:
        plt.xlim(-0.5, shape[0] - 0.5)
        plt.ylim(-0.5, shape[1] - 0.5)
    if title is not None:
        plt.title(title)


def spike_train_plot_yx(data: Union[np.ndarray, torch.Tensor], polarity: bool = True, show: bool = True, save: str = None, titles: Iterable[str] = None, figsize: Tuple = (8, 6)) -> None:
    """
    二维空间的脉冲序列数据打印。
    Args:
        data (np.ndarray | torch.Tensor):需要打印的数据（可以是2、3、4维）
        polarity (bool): 是否将第1个维度作为极性维度，如果为True，则传入的数据为[C(P), H, W]（3维）或[B, C(P), H, W]（4维）；如果为False，则传入的数据为[H, W]（2维）或[B, H, W]（3维）
        show (bool): 是否展示图像，默认展示
        save (str): 是否保存图像，若传入路径，则保存图像；否则不保存
        titles (str*): 每张图都是什么标题
        figSize (Tuple): 图像大小
    """
    get_title = lambda b: titles[b] if titles is not None and b >= 0 and b < len(titles) else ("Event Group %d" % (b,))
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    dim = data.ndim
    indices = np.array(data.nonzero())
    fig = plt.figure(figsize = figsize)
    if dim == 2: # [H, W]
        event_plot_2d(
            indices[[1, 0]].T,
            values_pos = data[indices[0], indices[1]],
            color_pos = "#00ff00",
            title = get_title(0),
            index_0 = "x",
            index_1 = "y",
            shape = (data.shape[1], data.shape[0])
        )
    if dim == 3:
        if polarity: # [C(P), H, W]
            indices_pos = indices[:, indices[0] == 1][[1, 2]]
            indices_neg = indices[:, indices[0] == 0][[1, 2]]
            event_plot_2d(
                indices_pos[[1, 0]].T,
                indices_neg[[1, 0]].T,
                values_pos = data[1, indices_pos[0], indices_pos[1]],
                values_neg = data[0, indices_neg[0], indices_neg[1]],
                title = get_title(0),
                index_0 = "x",
                index_1 = "y",
                shape = (data.shape[2], data.shape[1])
            )
        else: # [B, H, W]
            batch_size = data.shape[0]
            rows = batch_size
            cols = 1
            if not rows % 2:
                rows = rows // 2
                cols = cols * 2
            for b in range(batch_size):
                indices_this_batch = indices[:, indices[0] == b][[1, 2]]
                plt.subplot(rows, cols, b + 1)
                event_plot_2d(
                    indices_this_batch[[1, 0]].T,
                    values_pos = data[b, indices_this_batch[0], indices_this_batch[1]],
                    color_pos = "#00ff00",
                    title = get_title(b),
                    index_0 = "x",
                    index_1 = "y",
                    shape = (data.shape[2], data.shape[1])
                )
    if dim == 4: # [B, C(P), H, W]
        batch_size = data.shape[0]
        rows = batch_size
        cols = 1
        if not rows % 2:
            rows = rows // 2
            cols = cols * 2
        for b in range(batch_size):
            indices_this_batch = indices[:, indices[0] == b][[1, 2, 3]]
            indices_pos = indices_this_batch[:, indices_this_batch[0] == 1][[1, 2]]
            indices_neg = indices_this_batch[:, indices_this_batch[0] == 0][[1, 2]]
            plt.subplot(rows, cols, b + 1)
            event_plot_2d(
                indices_pos[[1, 0]].T,
                indices_neg[[1, 0]].T,
                values_pos = data[b, 1, indices_pos[0], indices_pos[1]],
                values_neg = data[b, 0, indices_neg[0], indices_neg[1]],
                title = get_title(b),
                index_0 = "x",
                index_1 = "y",
                shape = (data.shape[3], data.shape[2]),
            )
    if show:
        fig.canvas.manager.set_window_title("Event Plotter Result")
        plt.show()
    if save is not None:
        plt.savefig(save)
        plt.close()


def event_seq_plot_yx(data: Union[np.ndarray, torch.Tensor], shape: Iterable, show: bool = True, save: str = None, titles: Iterable[str] = None, figsize: Tuple = (8, 6)) -> None:
    """
    二维空间的事件序列数据打印。
    Args:
        data (np.ndarray | torch.Tensor): 需要打印的数据（可以是2、3维）
        shape (Iterable): 图的形状（长度为data列数的元组），若不指定（或为0）则为数据的长度
        show (bool): 是否展示图像，默认展示
        save (str): 是否保存图像，若传入路径，则保存图像；否则不保存
        titles (str*): 每张图都是什么标题
        figSize (Tuple): 图像大小
    """
    get_title = lambda b: titles[b] if titles is not None and b >= 0 and b < len(titles) else ("Event Group %d" % (b,))
    get_shape = lambda i: shape[i] if shape is not None and len(shape) > i and shape[i] > 0 else (np.max(data[:, i]) - np.min(data[:, i]) + 1)
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    data, values = np.unique(data, axis = 0, return_counts = True)
    dim = data.shape[1]
    fig = plt.figure(figsize = figsize)
    if dim == 3: # [P, H, W]
        indices_pos = data[data[:, 0] == 1][:, [1, 2]]
        indices_neg = data[data[:, 0] == 0][:, [1, 2]]
        values_pos = values[data[:, 0] == 1]
        values_neg = values[data[:, 0] == 0]
        event_plot_2d(
            indices_pos[:, [1, 0]],
            indices_neg[:, [1, 0]],
            values_pos = values_pos,
            values_neg = values_neg,
            title = get_title(0),
            index_0 = "x",
            index_1 = "y",
            shape = (get_shape(2), get_shape(1))
        )
    elif dim == 2: # [H, W]
        event_plot_2d(
            data[:, [1, 0]],
            values_pos = values,
            color_pos = "#00ff00",
            title = get_title(0),
            index_0 = "x",
            index_1 = "y",
            shape = (get_shape(1), get_shape(0))
        )
    if show:
        fig.canvas.manager.set_window_title("Event Plotter Result")
        plt.show()
    if save is not None:
        plt.savefig(save)
        plt.close()


def event_plot_yx(data: Union[np.ndarray, torch.Tensor], shape: Iterable = None, polarity: bool = True, show: bool = True, save: str = None, titles: Iterable[str] = None, figsize: Tuple = (8, 6), is_seq: bool = False) -> None:
    """
    二维空间的事件数据打印。
    Args:
        data (np.ndarray | torch.Tensor): 需要打印的数据
        shape (Iterable): 图的形状（打印序列用）
        polarity (bool): 是否将第1个维度作为极性维度，如果为True，则传入的数据为[C(P), H, W]（3维）或[B, C(P), H, W]（4维）；如果为False，则传入的数据为[H, W]（2维）或[B, H, W]（3维）
        show (bool): 是否展示图像，默认展示
        save (str): 是否保存图像，若传入路径，则保存图像；否则不保存
        titles (str*): 每张图都是什么标题
        figSize (Tuple): 图像大小
        is_seq (bool): 是否为序列，True表明事件为形状为[n, 4]的序列，否则为形状为[T, C(P), H, W]的张量
    """
    if is_seq:
        event_seq_plot_yx(
            data = data,
            shape = shape,
            show = show,
            save = save,
            titles = titles,
            figsize = figsize
        )
    else:
        spike_train_plot_yx(
            data = data,
            polarity = polarity,
            show = show,
            save = save,
            titles = titles,
            figsize = figsize
        )


def spike_train_plot_tx(data: Union[np.ndarray, torch.Tensor], polarity: bool = True, show: bool = True, save: str = None, titles: Iterable[str] = None, figsize: Tuple = (8, 6)) -> None:
    """
    一维空间+一维时间的脉冲序列数据打印。
    Args:
        data (np.ndarray | torch.Tensor):需要打印的数据（可以是2、3、4维）
        polarity (bool): 是否将第2个维度作为极性维度，如果为True，则传入的数据为[T, C(P), L]（3维）或[B, T, C(P), L]（4维）；如果为False，则传入的数据为[T, L]（2维）或[B, T, L]（3维）
        show (bool): 是否展示图像，默认展示
        save (str): 是否保存图像，若传入路径，则保存图像；否则不保存
        titles (str*): 每张图都是什么标题
        figSize (Tuple): 图像大小
    """
    get_title = lambda b: titles[b] if titles is not None and b >= 0 and b < len(titles) else ("Event Group %d" % (b,))
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    dim = data.ndim
    indices = np.array(data.nonzero())
    fig = plt.figure(figsize = figsize)
    if dim == 2: # [T, L]
        event_plot_2d(
            indices[[0, 1]].T,
            values_pos = data[indices[0], indices[1]],
            color_pos = "#00ff00",
            title = get_title(0),
            shape = (data.shape[0], data.shape[1])
        )
        event_plot_2d(data, indices, color_pos = "#00ff00", title = get_title(0))
    if dim == 3:
        if polarity: # [T, C(P), L]
            indices_pos = indices[:, indices[1] == 1][[0, 2]]
            indices_neg = indices[:, indices[1] == 0][[0, 2]]
            event_plot_2d(
                indices_pos[[0, 1]].T,
                indices_neg[[0, 1]].T,
                values_pos = data[indices_pos[0], 1, indices_pos[1]],
                values_neg = data[indices_neg[0], 0, indices_neg[1]],
                title = get_title(0),
                shape = (data.shape[0], data.shape[2])
            )
        else: # [B, T, L]
            batch_size = data.shape[0]
            rows = batch_size
            cols = 1
            if not rows % 2:
                rows = rows // 2
                cols = cols * 2
            for b in range(batch_size):
                indices_this_batch = indices[:, indices[0] == b][[1, 2]]
                plt.subplot(rows, cols, b + 1)
                event_plot_2d(
                    indices_this_batch[[0, 1]].T,
                    values_pos = data[b, indices_this_batch[0], indices_this_batch[1]],
                    color_pos = "#00ff00",
                    title = get_title(b),
                    shape = (data.shape[1], data.shape[2])
                )
    if dim == 4: # [B, T, C(P), L]
        batch_size = data.shape[0]
        rows = batch_size
        cols = 1
        if not rows % 2:
            rows = rows // 2
            cols = cols * 2
        for b in range(batch_size):
            indices_this_batch = indices[:, indices[0] == b][[1, 2, 3]]
            indices_pos = indices_this_batch[:, indices_this_batch[1] == 1][[0, 2]]
            indices_neg = indices_this_batch[:, indices_this_batch[1] == 0][[0, 2]]
            plt.subplot(rows, cols, b + 1)
            event_plot_2d(
                indices_pos[[0, 1]].T,
                indices_neg[[0, 1]].T,
                values_pos = data[b, indices_pos[0], 1, indices_pos[1]],
                values_neg = data[b, indices_neg[0], 0, indices_neg[1]],
                title = get_title(b),
                shape = (data.shape[1], data.shape[3])
            )
    if show:
        fig.canvas.manager.set_window_title("Event Plotter Result")
        plt.show()
    if save is not None:
        plt.savefig(save)
        plt.close()


def event_seq_plot_tx(data: Union[np.ndarray, torch.Tensor], shape: Iterable, show: bool = True, save: str = None, titles: Iterable[str] = None, figsize: Tuple = (8, 6)) -> None:
    """
    一维空间+一维时间的事件序列数据打印。
    Args:
        data (np.ndarray | torch.Tensor): 需要打印的数据（可以是2、3维）
        shape (Iterable): 图的形状（长度为data列数的元组），若不指定（或为0）则为数据的长度
        show (bool): 是否展示图像，默认展示
        save (str): 是否保存图像，若传入路径，则保存图像；否则不保存
        titles (str*): 每张图都是什么标题
        figSize (Tuple): 图像大小
    """
    get_title = lambda b: titles[b] if titles is not None and b >= 0 and b < len(titles) else ("Event Group %d" % (b,))
    get_shape = lambda i: shape[i] if shape is not None and len(shape) > i and shape[i] > 0 else (np.max(data[:, i]) - np.min(data[:, i]) + 1)
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    data, values = np.unique(data, axis = 0, return_counts = True)
    dim = data.shape[1]
    fig = plt.figure(figsize = figsize)
    if dim == 3: # [T, C(P), L]
        indices_pos = data[data[:, 1] == 1][:, [0, 2]]
        indices_neg = data[data[:, 1] == 0][:, [0, 2]]
        values_pos = values[data[:, 1] == 1]
        values_neg = values[data[:, 1] == 0]
        event_plot_2d(
            indices_pos[:, [0, 1]],
            indices_neg[:, [0, 1]],
            values_pos = values_pos,
            values_neg = values_neg,
            title = get_title(0),
            shape = (get_shape(0), get_shape(2))
        )
    elif dim == 2: # [T, L]
        event_plot_2d(
            data[:, [0, 1]],
            values_pos = values,
            color_pos = "#00ff00",
            title = get_title(0),
            shape = (get_shape(0), get_shape(1))
        )
    if show:
        fig.canvas.manager.set_window_title("Event Plotter Result")
        plt.show()
    if save is not None:
        plt.savefig(save)
        plt.close()


def event_plot_tx(data: Union[np.ndarray, torch.Tensor], shape: Iterable = None, polarity: bool = True, show: bool = True, save: str = None, titles: Iterable[str] = None, figsize: Tuple = (8, 6), is_seq: bool = False) -> None:
    """
    一维空间+一维时间的事件数据打印。
    Args:
        data (np.ndarray | torch.Tensor): 需要打印的数据
        shape (Iterable): 图的形状（打印序列用）
        polarity (bool): 是否将第2个维度作为极性维度，如果为True，则传入的数据为[T, C(P), L]（3维）或[B, T, C(P), L]（4维）；如果为False，则传入的数据为[T, L]（2维）或[B, T, L]（3维）（打印张量用）
        show (bool): 是否展示图像，默认展示
        save (str): 是否保存图像，若传入路径，则保存图像；否则不保存
        titles (str*): 每张图都是什么标题
        figSize (Tuple): 图像大小
        is_seq (bool): 是否为序列，True表明事件为形状为[n, 4]的序列，否则为形状为[T, C(P), H, W]的张量
    """
    if is_seq:
        event_seq_plot_tx(
            data = data,
            shape = shape,
            show = show,
            save = save,
            titles = titles,
            figsize = figsize
        )
    else:
        spike_train_plot_tx(
            data = data,
            polarity = polarity,
            show = show,
            save = save,
            titles = titles,
            figsize = figsize
        )


def event_plot_3d(ax: Axes3D, indices_pos: np.ndarray, indices_neg: np.ndarray = None, values_pos: np.ndarray = None, values_neg: np.ndarray = None, color_pos: str = "#0000ff", color_neg: str = "#ff0000", title: str = None, index_0: str = "t", index_1: str = "y", index_2: str = "x", shape: Tuple = None, shift: float = 0.1) -> None:
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
        title (str): 图片的标题
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
    if title is not None:
        ax.set_title(title)


def spike_train_plot_tyx(data: Union[np.ndarray, torch.Tensor], polarity: bool = True, show: bool = True, save: str = None, titles: Iterable[str] = None, figsize: Tuple = (8, 6)) -> None:
    """
    二维空间+一维时间的脉冲序列数据打印。
    Args:
        data (np.ndarray | torch.Tensor):需要打印的数据（可以是3、4、5维）
        polarity (bool): 是否将第2个维度作为极性维度，如果为True，则传入的数据为[T, C(P), H, W]（4维）或[B, T, C(P), H, W]（5维）；如果为False，则传入的数据为[T, H, W]（3维）或[B, T, H, W]（4维）
        show (bool): 是否展示图像，默认展示
        save (str): 是否保存图像，若传入路径，则保存图像；否则不保存
        titles (str*): 每张图都是什么标题
        figSize (Tuple): 图像大小
    """
    get_title = lambda b: titles[b] if titles is not None and b >= 0 and b < len(titles) else ("Event Group %d" % (b,))
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    dim = data.ndim
    indices = np.array(data.nonzero())
    fig = plt.figure(figsize = figsize)
    if dim == 3: # [T, H, W]
        ax = fig.add_subplot(projection='3d')
        event_plot_3d(
            ax,
            indices[[0, 1, 2]].T,
            values_pos = data[indices[0], indices[1], indices[2]],
            color_pos = "#00ff00",
            title = get_title(0),
            shape = (data.shape[0], data.shape[1], data.shape[2])
        )
    if dim == 4:
        if polarity: # [T, C(P), H, W]
            ax = fig.add_subplot(projection='3d')
            indices_pos = indices[:, indices[1] == 1][[0, 2, 3]]
            indices_neg = indices[:, indices[1] == 0][[0, 2, 3]]
            event_plot_3d(
                ax,
                indices_pos[[0, 1, 2]].T,
                indices_neg[[0, 1, 2]].T,
                values_pos = data[indices_pos[0], 1, indices_pos[1], indices_pos[2]],
                values_neg = data[indices_neg[0], 0, indices_neg[1], indices_neg[2]],
                title = get_title(0),
                shape = (data.shape[0], data.shape[2], data.shape[3])
            )
        else: # [B, T, H, W]
            batch_size = data.shape[0]
            rows = batch_size
            cols = 1
            if not rows % 3:
                rows = rows // 3
                cols = cols * 3
            elif not rows % 2:
                rows = rows // 2
                cols = cols * 2
            for b in range(batch_size):
                indices_this_batch = indices[:, indices[0] == b][[1, 2, 3]]
                ax = fig.add_subplot(rows, cols, b + 1, projection='3d')
                event_plot_3d(
                    ax,
                    indices_this_batch[[0, 1, 2]].T,
                    values_pos = data[b, indices_this_batch[0], indices_this_batch[1], indices_this_batch[2]],
                    color_pos = "#00ff00",
                    title = get_title(b),
                    shape = (data.shape[1], data.shape[2], data.shape[3])
                )
                event_plot_3d(ax, data[b], indices[:, indices[0] == b][[1, 2, 3]], color_pos = "#00ff00", title = get_title(b))
    if dim == 5: # [B, T, C(P), H, W]
        batch_size = data.shape[0]
        rows = batch_size
        cols = 1
        if not rows % 3:
            rows = rows // 3
            cols = cols * 3
        elif not rows % 2:
            rows = rows // 2
            cols = cols * 2
        for b in range(batch_size):
            indices_this_batch = indices[:, indices[0] == b][[1, 2, 3, 4]]
            indices_pos = indices_this_batch[:, indices_this_batch[1] == 1][[0, 2, 3]]
            indices_neg = indices_this_batch[:, indices_this_batch[1] == 0][[0, 2, 3]]
            ax = fig.add_subplot(rows, cols, b + 1, projection='3d')
            event_plot_3d(
                ax,
                indices_pos[[0, 1, 2]].T,
                indices_neg[[0, 1, 2]].T,
                values_pos = data[b, indices_pos[0], 1, indices_pos[1], indices_pos[2]],
                values_neg = data[b, indices_neg[0], 0, indices_neg[1], indices_neg[2]],
                title = get_title(b),
                shape = (data.shape[1], data.shape[3], data.shape[4])
            )
    if show:
        fig.canvas.manager.set_window_title("Event Plotter Result")
        plt.show()
    if save is not None:
        plt.savefig(save)
        plt.close()


def event_seq_plot_tyx(data: Union[np.ndarray, torch.Tensor], shape: Iterable, show: bool = True, save: str = None, titles: Iterable[str] = None, figsize: Tuple = (8, 6)) -> None:
    """
    二维空间+一维时间的事件序列数据打印。
    Args:
        data (np.ndarray | torch.Tensor): 需要打印的数据（可以是3、4维）
        shape (Iterable): 图的形状（长度为data列数的元组），若不指定（或为0）则为数据的长度
        show (bool): 是否展示图像，默认展示
        save (str): 是否保存图像，若传入路径，则保存图像；否则不保存
        titles (str*): 每张图都是什么标题
        figSize (Tuple): 图像大小
    """
    get_title = lambda b: titles[b] if titles is not None and b >= 0 and b < len(titles) else ("Event Group %d" % (b,))
    get_shape = lambda i: shape[i] if shape is not None and len(shape) > i and shape[i] > 0 else (np.max(data[:, i]) - np.min(data[:, i]) + 1)
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    data, values = np.unique(data, axis = 0, return_counts = True)
    dim = data.shape[1]
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(projection='3d')
    if dim == 4: # [T, C(P), H, W]
        indices_pos = data[data[:, 1] == 1][:, [0, 2, 3]]
        indices_neg = data[data[:, 1] == 0][:, [0, 2, 3]]
        values_pos = values[data[:, 1] == 1]
        values_neg = values[data[:, 1] == 0]
        event_plot_3d(
            ax,
            indices_pos[:, [0, 1, 2]],
            indices_neg[:, [0, 1, 2]],
            values_pos = values_pos,
            values_neg = values_neg,
            title = get_title(0),
            shape = (get_shape(0), get_shape(2), get_shape(3))
        )
    elif dim == 3: # [T, H, W]
        event_plot_3d(
            ax,
            data[:, [0, 1, 2]],
            values_pos = values,
            color_pos = "#00ff00",
            title = get_title(0),
            shape = (get_shape(0), get_shape(1), get_shape(2))
        )
    if show:
        fig.canvas.manager.set_window_title("Event Plotter Result")
        plt.show()
    if save is not None:
        plt.savefig(save)
        plt.close()


def event_plot_tyx(data: Union[np.ndarray, torch.Tensor], shape: Iterable = None, polarity: bool = True, show: bool = True, save: str = None, titles: Iterable[str] = None, figsize: Tuple = (8, 6), is_seq: bool = False) -> None:
    """
    二维空间+一维时间的事件数据打印。
    Args:
        data (np.ndarray | torch.Tensor): 需要打印的数据
        shape (Iterable): 图的形状（打印序列用）
        polarity (bool): 是否将第2个维度作为极性维度，如果为True，则传入的数据为[T, C(P), H, W]（4维）或[B, T, C(P), H, W]（5维）；如果为False，则传入的数据为[T, H, W]（3维）或[B, T, H, W]（4维）（打印张量用）
        show (bool): 是否展示图像，默认展示
        save (str): 是否保存图像，若传入路径，则保存图像；否则不保存
        titles (str*): 每张图都是什么标题
        figSize (Tuple): 图像大小
        is_seq (bool): 是否为序列，True表明事件为形状为[n, 4]的序列，否则为形状为[T, C(P), H, W]的张量
    """
    if is_seq:
        event_seq_plot_tyx(
            data = data,
            shape = shape,
            show = show,
            save = save,
            titles = titles,
            figsize = figsize
        )
    else:
        spike_train_plot_tyx(
            data = data,
            polarity = polarity,
            show = show,
            save = save,
            titles = titles,
            figsize = figsize
        )


def graph_plot_by_adjacent(adjacent: Union[np.ndarray, torch.Tensor], show: bool = True, save: str = None, figsize: Tuple = (6, 6)) -> None:
    from pyecharts.charts import Graph
    from pyecharts.options import LineStyleOpts
    import webbrowser
    get_name = lambda i: "Neuron %d" % (i + 1,)
    fig = plt.figure(figsize = figsize)
    neuron_num = max(adjacent.shape[0], adjacent.shape[1])
    neurons = []
    for i in range(neuron_num):
        neurons.append({"name": get_name(i), "symbolSize": 10})
    synapses = []
    axons, dendrites = torch.nonzero(adjacent, as_tuple = True)
    axons, dendrites = axons.cpu().numpy().tolist(), dendrites.cpu().numpy().tolist()
    for i in range(len(axons)):
        synapses.append({"source": axons[i], "target": dendrites[i]})
    graph = Graph({
        "title": "LSM Graph",
        "width": "1920px",
        "height": "1080px",
        "is_animation": False
    })
    graph.add(
        "LSM Graph",
        neurons,
        synapses,
        layout = "circular",
        edge_length = 50,
        gravity = 0.2,
        repulsion = 50,
        edge_symbol = [None, "arrow"],
        linestyle_opts = LineStyleOpts(
            curve = 1.0
        ),
    )
    if save:
        graph.render(save)
    else:
        graph.render()
        save = os.path.join(os.getcwd(), "render.html")
    if show:
        webbrowser.open("file://" + save.replace("\\", "/"))