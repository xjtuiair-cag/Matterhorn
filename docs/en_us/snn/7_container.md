# `matterhorn_pytorch.snn.container`

[Back to `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/7_container.md)

[中文](../../zh_cn/snn/7_container.md)

## Module Introduction

This module serves as a container for the `matterhorn_pytorch.snn.Module`, combining various modules.

## `matterhorn_pytorch.snn.Sequential` / `matterhorn_pytorch.snn.container.Sequential`

SNN sequential container. Similar to `nn.Sequential`, but:

(1) It supports all `torch.nn.Module` modules. If a method of `matterhorn_pytorch.snn.Module` is applied on the `Sequential` module, it applies on all submodules that belong to `matterhorn_pytorch.snn.Module`.

(2) When the output of the inside layer is a tuple, the first element will be used as next layer's input by default, and the other elements will be kept as global history outputs.

```python
Sequential(
    *args: Tuple[nn.Module]
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
)
print(model)
```