import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Iterable


def event_plot_xyt(ax: Axes3D, indices_pos: np.ndarray, indices_neg: np.ndarray = None, color_pos: str = "#ff0000", color_neg: str = "#0000ff", title: str = None) -> None:
    """
    单次二维空间+一维时间的事件数据打印。
    @params:
        ax: plt.Axes 
        indices_pos: np.ndarray （正）事件索引
        indices_neg: np.ndarray 负事件索引，如果没有极性传入None
        color_pos: str 正事件颜色（Hex）
        color_neg: str 负事件颜色（Hex）
    """
    ax.scatter(indices_pos[2], indices_pos[0], indices_pos[1], s = 1, c = color_pos)
    if indices_neg is not None:
        ax.scatter(indices_neg[2], indices_neg[0], indices_neg[1], s = 1, c = color_neg)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("y")
    if title is not None:
        ax.set_title(title)


def event_plot_3d(data: torch.Tensor, polarity: bool = True, show: bool = True, save: str = None, titles: Iterable[str] = None) -> None:
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
    data = data.numpy()
    indices = np.array(data.nonzero())
    fig = plt.figure()
    if dim == 3: # [T, H, W]
        ax = fig.add_subplot(projection='3d')
        event_plot_xyt(ax, indices, title = get_title(0), color_pos = "#00ff00")
    if dim == 4:
        if polarity: # [T, C(P), H, W]
            ax = fig.add_subplot(projection='3d')
            event_plot_xyt(ax, indices[:, indices[1] == 1][[0, 2, 3]], indices[:, indices[1] == 0][[0, 2, 3]], title = get_title(0))
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
                ax = fig.add_subplot(rows, cols, b + 1, projection='3d')
                event_plot_xyt(ax, indices[:, indices[0] == b][[1, 2, 3]], title = get_title(b), color_pos = "#00ff00")
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
            ax = fig.add_subplot(rows, cols, b + 1, projection='3d')
            event_plot_xyt(ax, indices[:, (indices[0] == b) & (indices[2] == 1)][[1, 3, 4]], indices[:, (indices[0] == b) & (indices[2] == 0)][[1, 3, 4]], title = get_title(b))
    if show:
        fig.canvas.set_window_title("Event Plotter Result")
        plt.show()
    if save is not None:
        plt.savefig(save)