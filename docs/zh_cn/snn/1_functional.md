# `matterhorn_pytorch.snn.functional`

[回到 `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/1_functional.md)

[中文](../../zh_cn/snn/1_functional.md)

## 模块简介

该模块为 `matterhorn_pytorch.snn` 模块的函数库，存储所被调用的函数。

## `matterhorn_pytorch.snn.functional.to_spike_train`

以 $x \ge 0.5$ 为界，将值转为脉冲（有或无）。

```python
to_spike_train(
    x: torch.Tensor
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$ ，模拟值。

### 返回值

`o (torch.Tensor)` ：输出 $O$ ，脉冲序列。

### 示例用法

```python
import torch
import matterhorn.snn.functional as SF


x = torch.rand(2, 3)
print(x)
y = SF.to_spike_train(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.heaviside_rectangular`

Heaviside 阶跃函数：

$$u(x)=x \ge 0 ? 1 : 0$$

以矩形窗作为反向传播的梯度。您可以在参考文献 [1] 中找到关于它们的详细定义。

```python
heaviside_rectangular(
    x: torch.Tensor,
    a: float = 1.0
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$ ，电位值。

`a (float)` ：参数 $a$ 。

### 返回值

`o (torch.Tensor)` ：输出 $O$ ，脉冲序列。

### 示例用法

```python
import torch
import matterhorn.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.heaviside_rectangular(x, 1.0)
print(y)
```

## `matterhorn_pytorch.snn.functional.heaviside_polynomial`

Heaviside 阶跃函数：

$$u(x)=x \ge 0 ? 1 : 0$$

以三角形窗作为反向传播的梯度。您可以在参考文献 [1] 中找到关于它们的详细定义。

```python
heaviside_polynomial(
    x: torch.Tensor,
    a: float = 4.0
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$ ，电位值。

`a (float)` ：参数 $a$ 。

### 返回值

`o (torch.Tensor)` ：输出 $O$ ，脉冲序列。

### 示例用法

```python
import torch
import matterhorn.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.heaviside_polynomial(x, 4.0)
print(y)
```

## `matterhorn_pytorch.snn.functional.heaviside_sigmoid`

Heaviside 阶跃函数：

$$u(x)=x \ge 0 ? 1 : 0$$

以 Sigmoid 函数的梯度作为反向传播的梯度。您可以在参考文献 [1] 中找到关于它们的详细定义。

```python
heaviside_sigmoid(
    x: torch.Tensor,
    a: float = 0.25
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$ ，电位值。

`a (float)` ：参数 $a$ 。

### 返回值

`o (torch.Tensor)` ：输出 $O$ ，脉冲序列。

### 示例用法

```python
import torch
import matterhorn.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.heaviside_sigmoid(x, 0.25)
print(y)
```

## `matterhorn_pytorch.snn.functional.heaviside_gaussian`

Heaviside 阶跃函数：

$$u(x)=x \ge 0 ? 1 : 0$$

以高斯函数作为反向传播的梯度。您可以在参考文献 [1] 中找到关于它们的详细定义。

```python
heaviside_gaussian(
    x: torch.Tensor,
    a: float = 0.16
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$ ，电位值。

`a (float)` ：参数 $a$ 。

### 返回值

`o (torch.Tensor)` ：输出 $O$ ，脉冲序列。

### 示例用法

```python
import torch
import matterhorn.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.heaviside_gaussian(x, 0.16)
print(y)
```

## 参考文献

[1] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.