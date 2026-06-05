# `matterhorn_pytorch.util.plotter`

[Back to `matterhorn_pytorch.util`](./README.md)

[English](../../en_us/util/1_plotter.md)

[中文](../../zh_cn/util/1_plotter.md)

## Module Introduction

This module is based on `matplotlib` and plots event sequences or spike tensors. Events typically contain a time dimension and one or more spatial dimensions. For example, events from event cameras usually have 4 dimensions `[T, P, Y, X]`.

The high-level functions `event_plot_*` return a matplotlib `Axes` object. Call `matplotlib.pyplot.show()` to display the figure.

### Common Keyword Arguments

All `event_plot_*` functions accept these optional keyword arguments via `**kwargs`:

`color_pos (str)`: Color for positive events (hex string, default `"#ff00ff"` or `"#0000ff"`).

`color_neg (str)`: Color for negative events (hex string, default `"#ff0000"`).

`index_0`, `index_1`, `index_2 (str)`: Axis labels for each dimension.

`prec (int)`: Temporal precision for 1D plots (used by `event_plot_tx`).

`shift (float)`: Spatial offset for 2D/3D plots to separate polarities.

`ax (Axes)`: Existing matplotlib axes. If `None`, a new figure is created.

## `matterhorn_pytorch.util.plotter.event_plot_yx`

Plot events with 2D spatial information `Y`, `X` (no explicit time axis in the plot).

```python
event_plot_yx(
    data: torch.Tensor,
    shape: Iterable = None,
    is_seq: bool = False,
    ax: Axes = None,
    **kwargs
) -> Axes
```

### Parameters

`data (torch.Tensor)`: Spike train or event sequence. As a spike train: `[C(P), H, W]` (3D) or `[T, C(P), H, W]` (4D). As an event sequence (`is_seq=True`): `[N, 2]` or `[N, 3]`.

`shape (Iterable)`: Required when `is_seq=True`. Sizes of each column in the event sequence.

`is_seq (bool)`: If `True`, `data` is an event sequence; otherwise it is a spike train.

`ax (Axes)`: Matplotlib axes to draw on.

`**kwargs`: See Common Keyword Arguments above.

### Returns

`ax (Axes)`: The matplotlib axes containing the plot.

### Example Usage

(1) Plot spike train:

```python
import torch
import matterhorn_pytorch as mth


events = (torch.rand(2, 128, 128) > 0.9).float()
print(events.shape)
ax = mth.util.plotter.event_plot_yx(events)
```

(2) Plot event sequence:

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

Plot events with 1D spatial information `X` and temporal information `T`.

```python
event_plot_tx(
    data: torch.Tensor,
    shape: Iterable = None,
    is_seq: bool = False,
    ax: Axes = None,
    **kwargs
) -> Axes
```

### Parameters

`data (torch.Tensor)`: Spike train or event sequence. As a spike train: `[T, C(P), L]` (3D) or `[T, B, C(P), L]` (4D). As an event sequence (`is_seq=True`): `[N, 2]` or `[N, 3]`.

`shape (Iterable)`: Required when `is_seq=True`.

`is_seq (bool)`: Whether `data` is an event sequence.

`ax (Axes)`: Matplotlib axes to draw on.

`**kwargs`: See Common Keyword Arguments above.

### Returns

`ax (Axes)`: The matplotlib axes containing the plot.

### Example Usage

(1) Plot spike train:

```python
import torch
import matterhorn_pytorch as mth


events = (torch.rand(64, 2, 128) > 0.9).float()
print(events.shape)
ax = mth.util.plotter.event_plot_tx(events)
```

(2) Plot event sequence:

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

Plot events with 2D spatial information `Y`, `X` and temporal information `T`.

```python
event_plot_tyx(
    data: torch.Tensor,
    shape: Iterable = None,
    is_seq: bool = False,
    ax: Axes = None,
    **kwargs
) -> Axes
```

### Parameters

`data (torch.Tensor)`: Spike train or event sequence. As a spike train: `[T, C(P), H, W]` (4D) or `[B, T, C(P), H, W]` (5D). As an event sequence (`is_seq=True`): `[N, 3]` or `[N, 4]`.

`shape (Iterable)`: Required when `is_seq=True`.

`is_seq (bool)`: Whether `data` is an event sequence.

`ax (Axes)`: Matplotlib axes to draw on.

`**kwargs`: See Common Keyword Arguments above.

### Returns

`ax (Axes)`: The matplotlib axes containing the plot.

### Example Usage

(1) Plot spike train:

```python
import torch
import matterhorn_pytorch as mth


events = (torch.rand(1000, 2, 128, 128) > 0.9998).float()
print(events.shape)
ax = mth.util.plotter.event_plot_tyx(events)
```

(2) Plot event sequence:

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

## Lower-Level Helpers

| Function | Description |
|----------|-------------|
| `init_figure(ndim, **kwargs)` | Create a matplotlib `Figure` and `Axes` (2D or 3D). |
| `spike_train_plot_yx/tx/tyx` | Plot a spike train on existing or new axes. |
| `event_seq_plot_yx/tx/tyx` | Convert an event sequence to a spike train, then plot. |
| `event_plot_1d/2d/3d` | Draw a single event batch on a given axes. |
| `transition_color` | Interpolate between two hex colors. |
