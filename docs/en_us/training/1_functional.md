# `matterhorn_pytorch.training.functional`

[Back to `matterhorn_pytorch.training`](./README.md)

[English](../../en_us/training/1_functional.md)

[中文](../../zh_cn/training/1_functional.md)

## Module Introduction

This module contains functions for custom training in the `matterhorn_pytorch.training` module.

## `matterhorn_pytorch.training.functional.stdp`

Function for Spike Timing-Dependent Plasticity (STDP). It computes the weight update based on input spike train and output spike train.

```python
stdp(
    delta_weight: torch.Tensor,
    input_spike_train: torch.Tensor,
    output_spike_train: torch.Tensor,
    a_pos: float,
    tau_pos: float,
    a_neg: float,
    tau_neg: float
) -> torch.Tensor
```

### Parameters

`delta_weight (torch.Tensor)`: Weight increment matrix $\Delta w$, with size `[O, I]`.

`input_spike_train (torch.Tensor)`: Input spike train of the layer, with size `[T, B, I]`.

`output_spike_train (torch.Tensor)`: Output spike train of the layer, with size `[T, B, O]`.

`a_pos (float)`: STDP parameter $A_{+}$.

`tau_pos (float)`: STDP parameter $\tau_{+}$.

`a_neg (float)`: STDP parameter $A_{-}$.

`tau_neg (float)`: STDP parameter $\tau_{-}$.

### Returns

`delta_weight (torch.Tensor)`: Computed weight increment matrix $\Delta w$, with size `[O, I]`.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth
import matterhorn_pytorch.snn.functional as SF
from matterhorn_pytorch.training.functional import stdp


delta_weight = torch.zeros(4, 6, dtype = torch.float) # [O = 4, I = 6]
input_spike_train = SF.to_spike_train(torch.rand(8, 1, 6)) # [T = 8, B = 1, I = 6]
output_spike_train = SF.to_spike_train(torch.rand(8, 1, 4)) # [T = 8, B = 1, O = 4]
delta_weight = stdp(delta_weight, input_spike_train, output_spike_train, 0.2, 2, 0.2, 2)
print(delta_weight)
```