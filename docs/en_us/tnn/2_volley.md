# `matterhorn_pytorch.tnn.volley`

[Back to `matterhorn_pytorch.tnn`](./README.md)

[English](../../en_us/tnn/2_volley.md)

[中文](../../zh_cn/tnn/2_volley.md)

## Module Introduction

This module implements **volley** encoding/decoding and space-time transforms for TNNs. In Smith's TNN framework, each input line is represented by the **first-spike timestamp** $t_i$; multiple lines form a **time volley**. Lines without spikes are denoted $\infty$ (implemented as `float("inf")` in code).

Reference: Smith J E. *Space-Time Computing with Temporal Neural Networks*, Section 6.2 (spike encoding) and Section 7.3 (OnOff encoding).

## `matterhorn_pytorch.tnn.EMPTY_SPIKE`

Sentinel constant for absent spikes, equal to `float("inf")`.

## `matterhorn_pytorch.tnn.is_spike_present`

Returns a boolean mask indicating which positions contain valid spikes.

```python
is_spike_present(t: torch.Tensor) -> torch.Tensor
```

### Arguments

`t (torch.Tensor)`: First-spike timestamps of arbitrary shape.

### Returns

`mask (torch.Tensor)`: Boolean mask; `True` means a spike is present.

## `matterhorn_pytorch.tnn.normalize_time_volley`

Normalizes a time volley to a local reference frame (earliest spike at 0). Corresponds to the **normalize** step in Smith Section 7.4.1 evaluation mode.

$$t_i^{norm} = t_i - \min_j\{t_j\}$$

```python
normalize_time_volley(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

### Arguments

`t (torch.Tensor)`: Time volley of shape `[..., M]`; absent lines use `EMPTY_SPIKE`.

### Returns

`t_norm (torch.Tensor)`: Normalized time volley.

`t_min (torch.Tensor)`: Global time of the earliest spike per volley.

### Example Usage

```python
import torch
import matterhorn_pytorch.tnn as TNN

t = torch.tensor([[1.0, 3.0, float("inf")]])
t_norm, t_min = TNN.normalize_time_volley(t)
print(t_norm)   # [[0.0, 2.0, inf]]
print(t_min)    # [1.0]
```

## `matterhorn_pytorch.tnn.denormalize_time_volley`

Restores a normalized volley to the global time reference.

$$t_i = t_i^{norm} + t_{min}$$

```python
denormalize_time_volley(t_norm: torch.Tensor, t_min: torch.Tensor) -> torch.Tensor
```

### Arguments

`t_norm (torch.Tensor)`: Normalized volley, shape `[..., M]`.

`t_min (torch.Tensor)`: Global earliest-spike time, shape `[...]` or `[..., 1]`.

### Returns

`t (torch.Tensor)`: Global time volley.

## `matterhorn_pytorch.tnn.value_to_time_volley`

Converts a value volley to a time volley:

$$v_i = \alpha(T_{max} - t_i), \quad \alpha = \frac{V_{max}}{T_{max}}$$

```python
value_to_time_volley(v: torch.Tensor, t_max: float, v_max: float) -> torch.Tensor
```

### Arguments

`v (torch.Tensor)`: Value volley, shape `[..., M]`; absent values use `EMPTY_SPIKE`.

`t_max (float)`: Maximum time window $T_{max}$.

`v_max (float)`: Maximum value $V_{max}$.

### Returns

`t (torch.Tensor)`: Time volley.

## `matterhorn_pytorch.tnn.time_to_value_volley`

Inverse of `value_to_time_volley`.

```python
time_to_value_volley(t: torch.Tensor, t_max: float, v_max: float) -> torch.Tensor
```

### Arguments

`t (torch.Tensor)`: Time volley.

`t_max (float)`: $T_{max}$.

`v_max (float)`: $V_{max}$.

### Returns

`v (torch.Tensor)`: Value volley.

## `matterhorn_pytorch.tnn.merge_time_volleys`

Concatenates multiple time volleys along the last dimension (CC input bundle merge).

```python
merge_time_volleys(*volleys: torch.Tensor) -> torch.Tensor
```

### Arguments

`*volleys (torch.Tensor)`: Time volleys sharing the same batch dimension.

### Returns

`merged (torch.Tensor)`: Merged time volley.

## `matterhorn_pytorch.tnn.onoff_encode`

OnOff encoding: concatenates each feature with its negative copy to remove inter-volley ordering bias (Smith Section 7.3):

$$x_{m+k} = I_{max} - x_k$$

```python
onoff_encode(values: torch.Tensor, v_max: float) -> torch.Tensor
```

### Arguments

`values (torch.Tensor)`: Raw features, shape `[..., M]`, in $[0, v_{max}]$.

`v_max (float)`: Maximum value $I_{max}/V_{max}$.

### Returns

`encoded (torch.Tensor)`: OnOff-encoded vector, shape `[..., 2M]`.

## `matterhorn_pytorch.tnn.OnOffEncoder`

`nn.Module` wrapper for OnOff encoding.

```python
OnOffEncoder(v_max: float = 1.0)
```

### Constructor Arguments

`v_max (float)`: Feature maximum $V_{max}/I_{max}$.

### Overridable Methods

#### `forward(self, x: torch.Tensor) -> torch.Tensor`

`x (torch.Tensor)`: Input features, shape `[B, M]`.

Returns: OnOff-encoded features, shape `[B, 2M]`.

### Example Usage

```python
import torch
from matterhorn_pytorch.tnn import OnOffEncoder, value_to_time_volley

encoder = OnOffEncoder(v_max=4.0)
values = torch.tensor([[1.0, 3.0]])
encoded = encoder(values)
times = value_to_time_volley(encoded, t_max=16.0, v_max=4.0)
```
