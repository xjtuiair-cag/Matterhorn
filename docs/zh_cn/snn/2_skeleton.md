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

### 私有方法

#### `_fold_for_parallel(self, x: torch.Tensor, target_dim: int | None = None) -> Tuple[torch.Tensor, int*]`

将数据的前几个维度压缩，使 `x.ndim == target_dim`，常用于压缩时间维度至批维度，使模块内能够执行多时间步的并行计算。

**参数**

`x (torch.Tensor)` ：未被压缩的张量。

`target_dim (int | None)` ：压缩到几维，为 `None` 代表不论几维，默认压缩前两个维度。

**返回值**

`y (torch.Tensor)` ：压缩后的张量。

`shape (int*)` ：被压缩的维度（第 0 维）原本的形状。

#### `_unfold_from_parallel(self, x: torch.Tensor, shape: int*) -> torch.Tensor`

并行计算后，将为并行计算而压缩的维度解压，使之重新回到正确的形状。

**参数**

`x (torch.Tensor)` ：被压缩的张量。

`shape (int*)` ：被压缩的维度（第 0 维）原本的形状。

**返回值**

`y (torch.Tensor)` ：解压缩后的张量。