# `matterhorn_pytorch.tnn.inhibition`

[Back to `matterhorn_pytorch.tnn`](./README.md)

[English](../../en_us/tnn/4_inhibition.md)

[中文](../../zh_cn/tnn/4_inhibition.md)

## Module Introduction

This module implements **inhibitory column** models from Smith Section 7.5. Inhibitory columns operate at a higher abstraction level: they filter volleys rather than modeling individual inhibitory neurons.

A cognitive column uses two inhibition mechanisms:

- **Feedforward Inhibition (FFI)**: suppresses overly dense input volleys;
- **Lateral Inhibition (LI)**: tk-WTA, keeping the earliest $k_L$ spikes within a time window.

Both operate in the **normalized time reference frame**.

## `matterhorn_pytorch.tnn.feedforward_inhibit`

Feedforward inhibition FFI (Smith Section 7.5.3). If more than $k_F$ spikes occur within $t_F$ after the first spike, the entire volley is nullified.

```python
feedforward_inhibit(
    t_volley: torch.Tensor,
    t_f: float,
    k_f: int
) -> torch.Tensor
```

### Arguments

`t_volley (torch.Tensor)`: Normalized time volley `[..., M]`.

`t_f (float)`: Window parameter $t_F$.

`k_f (int)`: Spike-count threshold $k_F$.

### Returns

`out (torch.Tensor)`: Filtered time volley.

### Example Usage

```python
import torch
from matterhorn_pytorch.tnn import feedforward_inhibit

t = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
out = feedforward_inhibit(t, t_f=1.3, k_f=10)  # entire volley cleared
```

## `matterhorn_pytorch.tnn.lateral_inhibit`

Lateral inhibition LI / tk-WTA (Smith Section 7.5.2). Keeps the earliest $k_L$ spikes with time $\le t_L$; others are cleared.

```python
lateral_inhibit(
    t_volley: torch.Tensor,
    t_l: float,
    k_l: int
) -> torch.Tensor
```

### Arguments

`t_volley (torch.Tensor)`: Normalized time volley `[..., M]`.

`t_l (float)`: Window $t_L$.

`k_l (int)`: Number of spikes to keep $k_L$.

### Returns

`out (torch.Tensor)`: Filtered time volley.

## `matterhorn_pytorch.tnn.FeedforwardInhibition`

FFI module wrapper.

```python
FeedforwardInhibition(t_f: float = 1.3, k_f: int = 10)
```

### Constructor Arguments

`t_f (float)`: Window $t_F$.

`k_f (int)`: Threshold $k_F$.

### Overridable Methods

#### `forward(self, t_volley: torch.Tensor, normalized: bool = True) -> torch.Tensor`

`t_volley (torch.Tensor)`: Time volley `[B, M]`.

`normalized (bool)`: If `False`, normalizes internally and denormalizes on output.

Returns: Filtered time volley.

## `matterhorn_pytorch.tnn.LateralInhibition`

LI module wrapper.

```python
LateralInhibition(t_l: float = 5.0, k_l: int = 3)
```

### Constructor Arguments

`t_l (float)`: Window $t_L$.

`k_l (int)`: Spikes to keep $k_L$.

### Overridable Methods

#### `forward(self, t_volley: torch.Tensor, normalized: bool = True) -> torch.Tensor`

Same arguments and return value as `FeedforwardInhibition.forward`.
