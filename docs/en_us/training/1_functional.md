# `matterhorn_pytorch.training.functional`

[Back to `matterhorn_pytorch.training`](./README.md)

[English](../../en_us/training/1_functional.md)

[中文](../../zh_cn/training/1_functional.md)

## Module Introduction

This module contains STDP-related functions for custom training in `matterhorn_pytorch.training`. They are used by `STDPLinear` and `STDPConv2d` in `matterhorn_pytorch.snn.layer`.

## `matterhorn_pytorch.training.functional.stdp_online`

Online Spike Timing-Dependent Plasticity (STDP) update for a single time step. It updates weight increments and input/output traces based on the current input and output spikes.

```python
stdp_online(
    delta_weight: torch.Tensor,
    input_trace: torch.Tensor,
    output_trace: torch.Tensor,
    input_spike_train: torch.Tensor,
    output_spike_train: torch.Tensor,
    a_pos: float,
    tau_pos: float,
    a_neg: float,
    tau_neg: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

### Parameters

`delta_weight (torch.Tensor)`: Weight increment matrix $\Delta w$, with size `[O, I]`.

`input_trace (torch.Tensor)`: Input trace, with size `[B, O, I]`.

`output_trace (torch.Tensor)`: Output trace, with size `[B, O, I]`.

`input_spike_train (torch.Tensor)`: Input spikes at the current time step, with size `[B, I]`.

`output_spike_train (torch.Tensor)`: Output spikes at the current time step, with size `[B, O]`.

`a_pos (float)`: STDP parameter $A_{+}$.

`tau_pos (float)`: STDP parameter $\tau_{+}$.

`a_neg (float)`: STDP parameter $A_{-}$.

`tau_neg (float)`: STDP parameter $\tau_{-}$.

### Returns

`delta_weight (torch.Tensor)`: Updated weight increment matrix, with size `[O, I]`.

`input_trace (torch.Tensor)`: Updated input trace, with size `[B, O, I]`.

`output_trace (torch.Tensor)`: Updated output trace, with size `[B, O, I]`.

## `matterhorn_pytorch.training.functional.stdp_linear`

STDP-based fully connected layer. It applies synaptic linear transformation, soma dynamics, and STDP weight updates.

```python
stdp_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    soma: Callable,
    hidden: Optional[torch.Tensor] = None,
    input_trace: Optional[torch.Tensor] = None,
    output_trace: Optional[torch.Tensor] = None,
    a_pos: float = 0.015,
    tau_pos: float = 2.0,
    a_neg: float = 0.015,
    tau_neg: float = 2.0,
    training: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

### Parameters

`input (torch.Tensor)`: Input spike train, with size `[T, B, I]`.

`weight (torch.Tensor)`: Weight matrix, with size `[O, I]`.

`soma (Callable)`: Soma module (e.g. `snn.IF`) that maps membrane potential to spikes.

`hidden (torch.Tensor | None)`: Historical membrane potential of the soma.

`input_trace (torch.Tensor | None)`: Input trace. Initialized to zeros if `None`.

`output_trace (torch.Tensor | None)`: Output trace. Initialized to zeros if `None`.

`a_pos`, `tau_pos`, `a_neg`, `tau_neg (float)`: STDP parameters.

`training (bool)`: Whether to compute STDP weight updates.

### Returns

`output (torch.Tensor)`: Output spike train.

`hidden (torch.Tensor)`: Updated membrane potential.

`input_trace (torch.Tensor)`: Updated input trace.

`output_trace (torch.Tensor)`: Updated output trace.

## `matterhorn_pytorch.training.functional.stdp_conv2d`

STDP-based 2D convolution layer. Same interface as `stdp_linear`, with additional convolution parameters.

```python
stdp_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    soma: Callable,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    hidden: Optional[torch.Tensor] = None,
    input_trace: Optional[torch.Tensor] = None,
    output_trace: Optional[torch.Tensor] = None,
    a_pos: float = 0.015,
    tau_pos: float = 2.0,
    a_neg: float = 0.015,
    tau_neg: float = 2.0,
    training: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

### Parameters

`input (torch.Tensor)`: Input spike train, with size `[T, B, CI, HI, WI]`.

`weight (torch.Tensor)`: Convolution kernel, with size `[CO, CI, KH, KW]`.

`soma (Callable)`: Soma module.

`stride (Tuple[int, int])`: Convolution stride.

`padding (Tuple[int, int])`: Convolution padding.

`dilation (Tuple[int, int])`: Convolution dilation.

Other parameters are the same as `stdp_linear`.

### Returns

Same as `stdp_linear`.

### Example Usage

```python
import torch
import matterhorn_pytorch.snn as snn
from matterhorn_pytorch.training.functional import stdp_online

delta_weight = torch.zeros(4, 6)
input_trace = torch.zeros(1, 4, 6)
output_trace = torch.zeros(1, 4, 6)
delta_weight, input_trace, output_trace = stdp_online(
    delta_weight,
    input_trace,
    output_trace,
    torch.randint(0, 2, (1, 6)).float(),
    torch.randint(0, 2, (1, 4)).float(),
    0.2, 2.0, 0.2, 2.0
)
print(delta_weight)
```
