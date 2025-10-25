# `matterhorn_pytorch.snn.skeleton`

[回到 `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/2_skeleton.md)

[中文](../../zh_cn/snn/2_skeleton.md)

## 模块简介

在该模块中定义了所有 SNN 模块的基类 `matterhorn_pytorch.snn.Module` 。其中 SNN 模块特有的变量与方法均在新定义的类中。 `matterhorn_pytorch.snn.Module` 继承了 `torch.nn.Module` ，并与 `torch.nn.Module` 的用法几乎一致。

## `matterhorn_pytorch.snn.Module` / `matterhorn_pytorch.snn.skeleton.Module`

```python
Module()
```

### 示例用法

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

### 可重载的方法

#### `extra_repr(self) -> str`

用法同 `torch.nn.Module.extra_repr()` ，自定义打印含有模块参数的字符串。