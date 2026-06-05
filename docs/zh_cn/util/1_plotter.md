# `matterhorn_pytorch.util.plotter`

[回到 `matterhorn_pytorch.util`](./README.md)

[English](../../en_us/util/1_plotter.md)

[中文](../../zh_cn/util/1_plotter.md)

## 模块简介

该模块基于 `matplotlib` ，将事件序列或脉冲张量绘制为图片。事件通常含有一个时间维度和多个空间维度。例如，事件相机所产生的事件通常含有 4 个维度 `[T, P, Y, X]` 。

高层函数 `event_plot_*` 返回 matplotlib 的 `Axes` 对象。调用 `matplotlib.pyplot.show()` 可显示图像。

### 通用关键字参数

所有 `event_plot_*` 函数通过 `**kwargs` 接受以下可选参数：

`color_pos (str)`：正事件颜色（十六进制字符串，默认 `"#ff00ff"` 或 `"#0000ff"`）。

`color_neg (str)`：负事件颜色（十六进制字符串，默认 `"#ff0000"`）。

`index_0`、`index_1`、`index_2 (str)`：各维度的坐标轴标签。

`prec (int)`：一维图的时间精度（用于 `event_plot_tx`）。

`shift (float)`：二维/三维图中分离极性的空间偏移。

`ax (Axes)`：已有的 matplotlib 坐标轴。为 `None` 时自动创建新图。

## `matterhorn_pytorch.util.plotter.event_plot_yx`

打印含有二维空间信息 `Y`、`X` 的事件（图中不显式展示时间轴）。

```python
event_plot_yx(
    data: torch.Tensor,
    shape: Iterable = None,
    is_seq: bool = False,
    ax: Axes = None,
    **kwargs
) -> Axes
```

### 参数

`data (torch.Tensor)`：脉冲序列或事件序列。作为脉冲序列：`[C(P), H, W]`（3 维）或 `[T, C(P), H, W]`（4 维）。作为事件序列（`is_seq=True`）：`[N, 2]` 或 `[N, 3]`。

`shape (Iterable)`：当 `is_seq=True` 时必须提供，为事件序列各列对应的尺寸。

`is_seq (bool)`：为 `True` 表示 `data` 是事件序列，否则为脉冲序列。

`ax (Axes)`：用于绘图的 matplotlib 坐标轴。

`**kwargs`：见上文通用关键字参数。

### 返回值

`ax (Axes)`：包含绘图的 matplotlib 坐标轴。

### 示例用法

（1）打印脉冲序列：

```python
import torch
import matterhorn_pytorch as mth


events = (torch.rand(2, 128, 128) > 0.9).float()
print(events.shape)
ax = mth.util.plotter.event_plot_yx(events)
```

（2）打印事件序列：

```python
import torch
import matterhorn_pytorch as mth


events = torch.zeros(1000, 3, dtype = torch.long)
events[:, 0] = (torch.rand(1000) > 0.5).long()
events[:, 1] = torch.randint(80, (1000,))
events[:, 2] = torch.randint(128, (1000,))
print(events.shape)
ax = mth.util.plotter.event_plot_yx(events, (2, 80, 128), is_seq = True)
```

## `matterhorn_pytorch.util.plotter.event_plot_tx`

打印含有一维空间信息 `X` 和一维时间信息 `T` 的事件。

```python
event_plot_tx(
    data: torch.Tensor,
    shape: Iterable = None,
    is_seq: bool = False,
    ax: Axes = None,
    **kwargs
) -> Axes
```

### 参数

`data (torch.Tensor)`：脉冲序列或事件序列。作为脉冲序列：`[T, C(P), L]`（3 维）或 `[T, B, C(P), L]`（4 维）。作为事件序列（`is_seq=True`）：`[N, 2]` 或 `[N, 3]`。

`shape (Iterable)`：当 `is_seq=True` 时必须提供。

`is_seq (bool)`：`data` 是否为事件序列。

`ax (Axes)`：用于绘图的 matplotlib 坐标轴。

`**kwargs`：见上文通用关键字参数。

### 返回值

`ax (Axes)`：包含绘图的 matplotlib 坐标轴。

### 示例用法

（1）打印脉冲序列：

```python
import torch
import matterhorn_pytorch as mth


events = (torch.rand(64, 2, 128) > 0.9).float()
print(events.shape)
ax = mth.util.plotter.event_plot_tx(events)
```

（2）打印事件序列：

```python
import torch
import matterhorn_pytorch as mth


events = torch.zeros(1000, 3, dtype = torch.long)
events[:, 0] = torch.randint(1600, (1000,))
events[:, 1] = (torch.rand(1000) > 0.5).long()
events[:, 2] = torch.randint(128, (1000,))
print(events.shape)
ax = mth.util.plotter.event_plot_tx(events, (0, 2, 128), is_seq = True)
```

## `matterhorn_pytorch.util.plotter.event_plot_tyx`

打印含有二维空间信息 `Y`、`X` 和一维时间信息 `T` 的事件。

```python
event_plot_tyx(
    data: torch.Tensor,
    shape: Iterable = None,
    is_seq: bool = False,
    ax: Axes = None,
    **kwargs
) -> Axes
```

### 参数

`data (torch.Tensor)`：脉冲序列或事件序列。作为脉冲序列：`[T, C(P), H, W]`（4 维）或 `[B, T, C(P), H, W]`（5 维）。作为事件序列（`is_seq=True`）：`[N, 3]` 或 `[N, 4]`。

`shape (Iterable)`：当 `is_seq=True` 时必须提供。

`is_seq (bool)`：`data` 是否为事件序列。

`ax (Axes)`：用于绘图的 matplotlib 坐标轴。

`**kwargs`：见上文通用关键字参数。

### 返回值

`ax (Axes)`：包含绘图的 matplotlib 坐标轴。

### 示例用法

（1）打印脉冲序列：

```python
import torch
import matterhorn_pytorch as mth


events = (torch.rand(1000, 2, 128, 128) > 0.9998).float()
print(events.shape)
ax = mth.util.plotter.event_plot_tyx(events)
```

（2）打印事件序列：

```python
import torch
import matterhorn_pytorch as mth


events = torch.zeros(4000, 4, dtype = torch.long)
events[:, 0] = torch.randint(1600, (4000,))
events[:, 1] = (torch.rand(4000) > 0.5).long()
events[:, 2] = torch.randint(80, (4000,))
events[:, 3] = torch.randint(128, (4000,))
print(events.shape)
ax = mth.util.plotter.event_plot_tyx(events, (0, 2, 80, 128), is_seq = True)
```

## 底层辅助函数

| 函数 | 说明 |
|------|------|
| `init_figure(ndim, **kwargs)` | 创建 matplotlib 的 `Figure` 与 `Axes`（2D 或 3D）。 |
| `spike_train_plot_yx/tx/tyx` | 在已有或新建的坐标轴上绘制脉冲序列。 |
| `event_seq_plot_yx/tx/tyx` | 将事件序列转为脉冲序列后绘制。 |
| `event_plot_1d/2d/3d` | 在给定坐标轴上绘制单批事件。 |
| `transition_color` | 在两个十六进制颜色之间插值。 |
