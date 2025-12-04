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

### Private Methods

#### `_fold_for_parallel(self, x: torch.Tensor, target_dim: int | None = None) -> Tuple[torch.Tensor, int*]`

Compresses the first few dimensions of the data so that `x.ndim == target_dim`. Commonly used to compress the time dimension into the batch dimension, enabling the module to perform parallel computation across multiple time steps.

**Parameters**

`x (torch.Tensor)`: The uncompressed tensor.

`target_dim (int | None)`: The target number of dimensions after compression. If `None`, the first two dimensions are compressed by default, regardless of the original number of dimensions.

**Returns**

`y (torch.Tensor)`: The compressed tensor.

`shape (int*)`: The original shape of the compressed dimension (the first dimension).

#### `_unfold_from_parallel(self, x: torch.Tensor, shape: int*) -> torch.Tensor`

After parallel computation, decompresses the dimension that was compressed for parallel computation, restoring it to its correct shape.

**Parameters**

`x (torch.Tensor)`: The compressed tensor.

`shape (int*)`: The original shape of the compressed dimension (the first dimension).

**Returns**

`y (torch.Tensor)`: The decompressed tensor.