# `matterhorn_pytorch.snn.skeleton`

[Back to `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/2_skeleton.md)

[中文](../../zh_cn/snn/2_skeleton.md)

## Module Introduction

In this module, the base class `matterhorn_pytorch.snn.Module` for all SNN modules is defined. The variables and methods specific to SNN modules are defined in the newly defined class. `matterhorn_pytorch.snn.Module` inherits from `torch.nn.Module`, and its usage is almost identical to `torch.nn.Module`.

## `matterhorn_pytorch.snn.Module` / `matterhorn_pytorch.snn.skeleton.Module`

```python
Module()
```

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


class Demo(mth.snn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return x
```

### Overridable Methods

#### `extra_repr(self) -> str`

Same usage as `torch.nn.Module.extra_repr()`, customizes the string containing module parameters for printing.