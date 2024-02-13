# `matterhorn_pytorch.snn.skeleton`

[Back to `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/2_skeleton.md)

[中文](../../zh_cn/snn/2_skeleton.md)

## Module Introduction

In this module, the base class `matterhorn_pytorch.snn.Module` for all SNN modules is defined. The variables and methods specific to SNN modules are defined in the newly defined class. `matterhorn_pytorch.snn.Module` inherits from `torch.nn.Module`, and its usage is almost identical to `torch.nn.Module`.

## `matterhorn_pytorch.snn.Module` / `matterhorn_pytorch.snn.skeleton.Module`

```python
Module(
    multi_time_step: bool = False,
    reset_after_process: bool = False
)
```

### Constructor Parameters

`multi_time_step (bool)`: Whether to switch to multi-time-step mode. Multi-time-step mode allows passing a tensor with multiple time steps, enabling computation step by step. Since Matterhorn defines many algorithms to accelerate multi-time-step operations, its efficiency is higher than that of single-time-step mode.

`reset_after_process (bool)`: Whether to reset automatically after execution. If you set it to False, you need to manually call the `reset()` method to reset the entire model after the SNN model finishes execution.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


class Demo(mth.snn.Module):
    def __init__(self) -> None:
        super().__init__(
            multi_time_step = True,
            reset_after_process = True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return x
```

### Overridable Methods

#### `extra_repr(self) -> str`

Same usage as `torch.nn.Module.extra_repr()`, customizes the string containing module parameters for printing.

#### `reset(self) -> matterhorn_pytorch.snn.Module`

Defines how the entire model should be reset globally. By default, it calls the `reset()` function of all sub-modules under `matterhorn_pytorch.snn.Module`. After executing all time steps, you need to reset the model (such as resetting the membrane potential of all neurons to the resting potential). At this time, you need to call this `reset()` function.

#### `detach(self) -> matterhorn_pytorch.snn.Module`

Defines how the computational graph of the model should be detached. By default, it calls the `detach()` function of all sub-modules under this module `matterhorn_pytorch.snn.Module`. When the time step is too long, it is obviously not reasonable to save the computational graph of all time steps. At this time, you can use the `detach()` function to detach some variables from the computational graph (with some risks, please contact us promptly if you find bugs).

#### `step(self, *args, **kwargs) -> matterhorn_pytorch.snn.Module`

Defines custom training deployment for the module. When the module does not use backpropagation as the training method, you need to call this method externally to train the model. By default, it calls the `step()` function of all sub-modules under this module `matterhorn_pytorch.snn.Module`. You can pass parameters (such as accuracy) from the outside to customize the update of variables such as weights in the module.