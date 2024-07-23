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

（1）其仅接受来自 `matterhorn_pytorch.snn.Module` 的 SNN 模块。

（2）其会保持模块内的模型同为单步模型或多步模型。

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

```python
Temporal(
    module: nn.Module
)
```

### 构造函数参数

`module (nn.Module)` ：单时间步 SNN 模块。

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

（1）其可以接受任何 `torch.nn.Module` 模块，并且对这个模块应用 `matterhorn_pytorch.snn.Module` 独有的方法时，其仅会对其中的 `matterhorn_pytorch.snn.Module` 模块生效。

（2）其会保持模块内的模型同为单步模型或多步模型。

```python
Sequential(
    *args
)
```

### 构造函数参数

`*args (*nn.Module)` ：按空间顺序传入的各个模块。

### 示例用法

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

ANN 模型的套壳容器，将 ANN 模型看作单时间步模式的 SNN 模型，赋予 `matterhorn.snn` 模块中所特有的属性与方法。

```python
Agent(
    nn_module: nn.Module,
    force_spike_output: bool = False
)
```

### 构造函数参数

`nn_module (nn.Module)` ：ANN 模块。

`force_spike_output (bool)` ：是否强制脉冲输出。默认不强制。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


model = mth.snn.Agent(
    nn.Linear(784, 10)
)
print(model)
```