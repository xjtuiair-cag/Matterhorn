# `matterhorn_pytorch.snn.container`

[Back to `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/7_container.md)

[中文](../../zh_cn/snn/7_container.md)

## Module Introduction

This module serves as a container for the `matterhorn_pytorch.snn.Module`, combining various modules.

Before introducing this chapter, it is necessary to introduce the single-time-step SNN model and the multi-time-step SNN model.

### ANN Model / Single-time-step SNN Model

The single-time-step SNN model is similar to the ANN model and accepts input with shape `[B, ...]` (n-dimensional). The single-time-step SNN model processes one time step per input loop.

### Multi-time-step SNN Model

The multi-time-step SNN model asynchronously loops through time in the computer. It accepts input with shape `[T, B, ...]` (n + 1-dimensional), looping `T` time steps at a time.

## `matterhorn_pytorch.snn.Spatial` / `matterhorn_pytorch.snn.container.Spatial`

Spatial container, similar to `torch.nn.Sequential`, but:

(1) It supports only a compound of SNN modules that belongs to `matterhorn_pytorch.snn.Module`.

(2) It will keep the step mode of modules in it consistent.

```python
Spatial(
    *args
)
```

### Constructor Arguments

`*args (*nn.Module)`: Various modules passed in spatial order.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


model = mth.snn.Spatial(
    mth.snn.Linear(784, 10),
    mth.snn.LIF()
)
print(model)
```

## `matterhorn_pytorch.snn.Temporal` / `matterhorn_pytorch.snn.container.Temporal`

Temporal container:

(1) Wraps single-time-step SNN modules and implements multi-time-step behavior through time loops.

(2) It is itself a multi-time-step module, so it consumes one more dimension `T` than single-time-step modules, by default treating the first dimension as the time dimension.

```python
Temporal(
    module: nn.Module
)
```

### Constructor Arguments

`module (nn.Module)`: Single-time-step SNN module.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


model = mth.snn.Temporal(
    mth.snn.LIF()
)
print(model)
```

## `matterhorn_pytorch.snn.Sequential` / `matterhorn_pytorch.snn.container.Sequential`

SNN sequential container, a combination of `Spatial` and `Temporal` containers. Similar to `Spatial`, but:

(1) It supports all `torch.nn.Module` modules. If a method of `matterhorn_pytorch.snn.Module` is applied on the `Sequential` module, it applies on all submodules that belong to `matterhorn_pytorch.snn.Module`.

(2) It will keep the step mode of modules in it consistent.

```python
Sequential(
    *args
)
```

### Constructor Arguments

`*args (*nn.Module)`: Various modules passed in spatial order.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


model = mth.snn.Sequential(
    mth.snn.Linear(784, 10),
    mth.snn.LIF()
).multi_step_mode_()
print(model)
```

## `matterhorn_pytorch.snn.Agent` / `matterhorn_pytorch.snn.container.Agent`

The shell aiming to convert ANN model to SNN model with single step mode in the simplest way. The attributes and methods of `matterhorn.snn` modules will also be attached at the model.

```python
Agent(
    nn_module: nn.Module,
    force_spike_output: bool = False
)
```

### Constructor Arguments

`nn_module (nn.Module)` : ANN module.

`force_spike_output (bool)` : Whether to force the output as spike output. Default is `False`.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


model = mth.snn.Agent(
    nn.Linear(784, 10)
)
print(model)
```