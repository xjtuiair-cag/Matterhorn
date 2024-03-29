# `matterhorn_pytorch.snn.container`

[回到 `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/7_container.md)

[中文](../../zh_cn/snn/7_container.md)

## 模块简介

SNN 模块的容器，用于组合各个 `matterhorn_pytorch.snn.Module` 。

在介绍本章之前，首先需要介绍单时间步 SNN 模型和多时间步 SNN 模型。

### ANN 模型 / 单时间步 SNN 模型

单时间步 SNN 模型与 ANN 模型类似，接受形状为 `[B, ...]` （`n` 维）的输入。单时间步 SNN 模型一次输入循环一个时间步。

### 多时间步 SNN 模型

多时间步 SNN 模型异步地在计算机中作时间循环，其接受形状为 `[T, B, ...]` （`n + 1` 维）的输入，一次循环 `T` 个时间步。

## `matterhorn_pytorch.snn.Spatial` / `matterhorn_pytorch.snn.container.Spatial`

空间容器，与 `torch.nn.Sequential` 类似，然而：

（1）对于 ANN 模块，其会直接与相邻模块相接。

（2）对于单时间步 SNN 模块，其并不会自动转换成多时间步 SNN 模块，因此最好保证同一个 `Spatial` 中的模块均为单时间步模块或均为多时间步模块，否则可能会出现时间步相关问题。

（3）其不存在自动重置机制。因此对于存储电位等临时变量的模块，您可能需要调用 `reset()` 方法手动重置。

```python
Spatial(
    *args
)
```

### 构造函数参数

`*args (*nn.Module)` ：按空间顺序传入的各个模块。

### 示例用法

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

时间容器：

（1）包裹单时间步 SNN 模块，通过时间循环实现多时间步。

（2）其本身为多时间步模块，因此要比单时间步模块多消耗一个维度 `T` ，默认将第一个维度视作时间维度。

（3）其存在自动重置机制。

```python
Temporal(
    module: nn.Module,
    reset_after_process: bool = True
)
```

### 构造函数参数

`module (nn.Module)` ：单时间步 SNN 模块。

`reset_after_process (bool)` ：是否在执行完后自动重置，若为 `False` 则需要手动重置。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


model = mth.snn.Temporal(
    mth.snn.LIF()
)
print(model)
```

## `matterhorn_pytorch.snn.Sequential` / `matterhorn_pytorch.snn.container.Sequential`

SNN 序列容器，结合了 `Spatial` 容器与 `Temporal` 容器的产物。其与 `Spatial` 用法类似，然而：

（1）对于 ANN 模块，其会直接与相邻模块相接。

（2）其中的单时间步 SNN 模块会自动转换成多时间步 SNN 模块。转换模式为：如果其自身支持多时间步，则直接将其转为多时间步；否则，在其外部加上一层 `Temporal` 容器，使其成为多时间步 SNN 模块。其本身为多时间步模块，因此要比单时间步模块多消耗一个维度 `T` ，默认将第一个维度视作时间维度。

（3）其存在自动重置机制。

推荐使用 `Sequential` 作为连接 `matterhorn_pytorch.snn` 的容器。

```python
Sequential(
    *args,
    reset_after_process: bool = True
)
```

### 构造函数参数

`*args (*nn.Module)` ：按空间顺序传入的各个模块。

`reset_after_process (bool)` ：是否在执行完后自动重置，若为 `False` 则需要手动重置。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


model = mth.snn.Sequential(
    mth.snn.Linear(784, 10),
    mth.snn.LIF()
)
print(model)
```