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
import matterhorn_pytorch.snn.functional as SF


x = torch.rand(2, 3)
print(x)
y = SF.to_spike_train(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.heaviside_rectangular`

Heaviside 阶跃函数：

$$u(x):=x \ge 0 ? 1 : 0$$

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
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.heaviside_rectangular(x, 1.0)
print(y)
```

## `matterhorn_pytorch.snn.functional.heaviside_polynomial`

Heaviside 阶跃函数：

$$u(x):=x \ge 0 ? 1 : 0$$

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
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.heaviside_polynomial(x, 4.0)
print(y)
```

## `matterhorn_pytorch.snn.functional.heaviside_sigmoid`

Heaviside 阶跃函数：

$$u(x):=x \ge 0 ? 1 : 0$$

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
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.heaviside_sigmoid(x, 0.25)
print(y)
```

## `matterhorn_pytorch.snn.functional.heaviside_gaussian`

Heaviside 阶跃函数：

$$u(x):=x \ge 0 ? 1 : 0$$

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
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.heaviside_gaussian(x, 0.16)
print(y)
```

## `matterhorn_pytorch.snn.functional.lt`

基于阶跃函数 $u(t)$ 实现的“小于”算子，以高斯函数作为反向传播的梯度。

```python
lt(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$。

`y (torch.Tensor)` ：输入 $Y$。

### 返回值

`z (torch.Tensor)` ：输出 $Z:=(X<Y)?1:0$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
y = torch.randint(-2, 3, size = (2, 3))
print(x, y)
z = SF.lt(x, y)
print(z)
```

## `matterhorn_pytorch.snn.functional.le`

基于阶跃函数 $u(t)$ 实现的“小于等于”算子，以高斯函数作为反向传播的梯度。

```python
le(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$。

`y (torch.Tensor)` ：输入 $Y$。

### 返回值

`z (torch.Tensor)` ：输出 $Z:=(X \le Y)?1:0$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
y = torch.randint(-2, 3, size = (2, 3))
print(x, y)
z = SF.le(x, y)
print(z)
```

## `matterhorn_pytorch.snn.functional.gt`

基于阶跃函数 $u(t)$ 实现的“大于”算子，以高斯函数作为反向传播的梯度。

```python
gt(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$。

`y (torch.Tensor)` ：输入 $Y$。

### 返回值

`z (torch.Tensor)` ：输出 $Z:=(X>Y)?1:0$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
y = torch.randint(-2, 3, size = (2, 3))
print(x, y)
z = SF.gt(x, y)
print(z)
```

## `matterhorn_pytorch.snn.functional.ge`

基于阶跃函数 $u(t)$ 实现的“大于等于”算子，以高斯函数作为反向传播的梯度。

```python
ge(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$。

`y (torch.Tensor)` ：输入 $Y$。

### 返回值

`z (torch.Tensor)` ：输出 $Z:=(X \ge Y)?1:0$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
y = torch.randint(-2, 3, size = (2, 3))
print(x, y)
z = SF.ge(x, y)
print(z)
```

## `matterhorn_pytorch.snn.functional.eq`

基于阶跃函数 $u(t)$ 实现的“等于”算子，以高斯函数作为反向传播的梯度。

```python
eq(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$。

`y (torch.Tensor)` ：输入 $Y$。

### 返回值

`z (torch.Tensor)` ：输出 $Z:=(X=Y)?1:0$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
y = torch.randint(-2, 3, size = (2, 3))
print(x, y)
z = SF.eq(x, y)
print(z)
```

## `matterhorn_pytorch.snn.functional.ne`

基于阶跃函数 $u(t)$ 实现的“不等于”算子，以高斯函数作为反向传播的梯度。

```python
ne(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$。

`y (torch.Tensor)` ：输入 $Y$。

### 返回值

`z (torch.Tensor)` ：输出 $Z:=(X \neq Y)?1:0$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
y = torch.randint(-2, 3, size = (2, 3))
print(x, y)
z = SF.ne(x, y)
print(z)
```

## `matterhorn_pytorch.snn.functional.floor`

基于多阶阶跃函数实现的“向下取整”算子。

```python
floor(
    x: torch.Tensor
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$。

### 返回值

`y (torch.Tensor)` ：输出 $Y=\lfloor X \rfloor$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.floor(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.ceil`

基于多阶阶跃函数实现的“向上取整”算子。

```python
ceil(
    x: torch.Tensor
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$。

### 返回值

`y (torch.Tensor)` ：输出 $Y=\lceil X \rceil$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.ceil(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.round`

基于多阶阶跃函数实现的“四舍五入”算子。

```python
round(
    x: torch.Tensor
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输入 $X$。

### 返回值

`y (torch.Tensor)` ：输出 $Y=\lfloor X + 0.5 \rfloor$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.round(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.encode_poisson`

脉冲的泊松编码，假定每个时间步产生的脉冲数符合泊松分布，速率编码的一种。

```python
encode_poisson(
    x: torch.Tensor,
    precision: float = 1e-5,
    count: bool = True
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：模拟值 $X$，作为脉冲发放的概率。

`precision (float)` ：由于在泊松分布中 $\lambda > 0$，这个参数规定了 $\lambda$ 的下界。

`count (bool)` ：是否发送脉冲计数。

### 返回值

`y (torch.Tensor)` ：输出脉冲 $O^{0}$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.rand(2, 3)
print(x)
y = SF.encode_poisson(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.encode_temporal`

脉冲的时间编码，当当前时间步超过传入的时间步后，以概率 `prob` 发射脉冲。

```python
encode_temporal(
    x: torch.Tensor,
    time_steps: int,
    t_offset: int = 0,
    prob: float = 1.0
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：模拟值 $X$，作为开始发放脉冲的时间步。

`time_steps (int)` ：生成的总时间步数。

`t_offset (int)` ：当前时间步的偏移量，即从哪个时间步开始计数。

`prob (float)` ：当到达开始发放脉冲的时间步后，发送脉冲的概率。

### 返回值

`y (torch.Tensor)` ：输出脉冲 $O^{0}$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(0, 8, (2, 3))
print(x)
y = SF.encode_temporal(x, 8)
print(y)
```

## `matterhorn_pytorch.snn.functional.encode_binary`

脉冲的二进制相位编码，将整数以二进制形式编码为脉冲序列，遵循大端序（即越高的比特位越早被编码）。

```python
encode_binary(
    x: torch.Tensor,
    length: int = 8,
    repeat: int = 1
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：模拟值 $X$，$[0,2^{T}-1]$ 之间的整数。

`length (int)` ：时间步长 $T$，即位宽，例如 `length = 8` 表示 $[0,255]$ 之间的整数。

`repeat (int)` ：重复次数。

### 返回值

`y (torch.Tensor)` ：输出脉冲 $O^{0}$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(0, 256, (2, 3))
print(x)
y = SF.encode_binary(x, 8)
print(y)
```

## `matterhorn_pytorch.snn.functional.decode_sum_spike`

计算脉冲总和实现脉冲解码。

```python
decode_sum_spike(
    x: torch.Tensor
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输出脉冲 $O^{L}$。

### 返回值

`y (torch.Tensor)` ：脉冲总和 $Y=\sum_{t=0}^{T-1}O^{L}(t)$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(0, 2, size = (8, 2, 3)).float()
print(x)
y = SF.decode_sum_spike(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.decode_avg_spike`

计算脉冲均值实现脉冲解码。

```python
decode_avg_spike(
    x: torch.Tensor
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输出脉冲 $O^{L}$。

### 返回值

`y (torch.Tensor)` ：脉冲均值 $Y=\frac{\sum_{t=0}^{T-1}O^{L}(t)}{T}$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(0, 2, size = (8, 2, 3)).float()
print(x)
y = SF.decode_avg_spike(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.decode_min_time`

计算脉冲最早出现的时间实现脉冲解码。

```python
decode_min_time(
    x: torch.Tensor,
    t_offset: int = 0,
    empty_fill: float = -1
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输出脉冲 $O^{L}$。

`t_offset (int)` ：时间步偏移量 $t_{0}$。

`empty_fill (float)` ：如果脉冲序列均为 $0$，使用什么值进行填充，默认为 $-1$.

### 返回值

`y (torch.Tensor)` ：最早含有脉冲的时间步 $Y=min\{t_{i}|O^{l}(t_{i})=1\}$。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(0, 2, size = (8, 2, 3)).float()
print(x)
y = SF.decode_min_time(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.decode_avg_time`

计算脉冲出现的平均时间实现脉冲解码。

```python
decode_avg_time(
    x: torch.Tensor,
    t_offset: int = 0,
    empty_fill: float = -1
) -> torch.Tensor
```

### 参数

`x (torch.Tensor)` ：输出脉冲 $O^{L}$。

`t_offset (int)` ：时间步偏移量 $t_{0}$。

`empty_fill (float)` ：如果脉冲序列均为 $0$，使用什么值进行填充，默认为 $-1$.

### 返回值

`y (torch.Tensor)` ：脉冲的平均时间步 $Y=\frac{\sum_{i}{\{t_{i}|O^{l}(t_{i})=1\}}}{n}$，其中 $n=\sum_{i}{\{1|O^{l}(t_{i})=1\}}$ 为脉冲个数。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(0, 2, size = (8, 2, 3)).float()
print(x)
y = SF.decode_avg_time(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.if_neuron`

IF 神经元的多步函数。

```python
if_neuron(
    x: torch.Tensor,
    h: torch.Tensor,
    u_threshold: torch.Tensor,
    u_rest: torch.Tensor,
    firing: str = "heaviside",
    hard_reset: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]
```

### 参数

`x (torch.Tensor)` ：输入电位 $X^{l}$，形状为 `[T, B, ...]`。

`h (torch.Tensor)` ：初始历史电位 $H^{l}(0)$，形状为 `[B, ...]`。

`u_threshold (torch.Tensor)` ：阈电位。

`u_rest (torch.Tensor)` ：静息电位。

`firing (str)` ：脉冲函数类型，默认为 `heaviside`，即 Heaviside 阶跃函数。

`hard_reset (bool)` ：是选择硬归零置重置（True）还是减法重置（False）。

### 返回值

`o (torch.Tensor)` ：输出脉冲 $O^{l}$，形状为 `[T, B, ...]`。

`h (torch.Tensor)` ：最终历史电位 $H^{l}(T)$，形状为 `[B, ...]`。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.rand(8, 2, 3)
h = torch.rand(2, 3)
y, h = SF.if_neuron(x, h, torch.tensor(1.0), torch.tensor(0.0))
print(y)
print(h)
```

## `matterhorn_pytorch.snn.functional.lif_neuron`

LIF 神经元的多步函数。

```python
lif_neuron(
    x: torch.Tensor,
    h: torch.Tensor,
    u_threshold: torch.Tensor,
    u_rest: torch.Tensor,
    tau_m: torch.Tensor,
    firing: str = "heaviside",
    hard_reset: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]
```

### 参数

`x (torch.Tensor)` ：输入电位 $X^{l}$，形状为 `[T, B, ...]`。

`h (torch.Tensor)` ：初始历史电位 $H^{l}(0)$，形状为 `[B, ...]`。

`u_threshold (torch.Tensor)` ：阈电位。

`u_rest (torch.Tensor)` ：静息电位。

`tau_m (torch.Tensor)` ：神经元时间常数 $\tau_{m}$。

`firing (str)` ：脉冲函数类型，默认为 `heaviside`，即 Heaviside 阶跃函数。

`hard_reset (bool)` ：是选择硬归零置重置（True）还是减法重置（False）。

### 返回值

`o (torch.Tensor)` ：输出脉冲 $O^{l}$，形状为 `[T, B, ...]`。

`h (torch.Tensor)` ：最终历史电位 $H^{l}(T)$，形状为 `[B, ...]`。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.rand(8, 2, 3) * 2
h = torch.rand(2, 3)
y, h = SF.lif_neuron(x, h, torch.tensor(1.0), torch.tensor(0.0), torch.tensor(2.0))
print(y)
print(h)
```

## `matterhorn_pytorch.snn.functional.qif_neuron`

QIF 神经元的多步函数。

```python
qif_neuron(
    x: torch.Tensor,
    h: torch.Tensor,
    u_threshold: torch.Tensor,
    u_rest: torch.Tensor,
    tau_m: torch.Tensor,
    u_c: torch.Tensor,
    a_0: torch.Tensor,
    firing: str = "heaviside",
    hard_reset: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]
```

### 参数

`x (torch.Tensor)` ：输入电位 $X^{l}$，形状为 `[T, B, ...]`。

`h (torch.Tensor)` ：初始历史电位 $H^{l}(0)$，形状为 `[B, ...]`。

`u_threshold (torch.Tensor)` ：阈电位。

`u_rest (torch.Tensor)` ：静息电位。

`tau_m (torch.Tensor)` ：神经元时间常数 $\tau_{m}$。

`u_c (torch.Tensor)` ：参数 $u_{c}$。

`a_0 (torch.Tensor)` ：参数 $a_{0}$。

`firing (str)` ：脉冲函数类型，默认为 `heaviside`，即 Heaviside 阶跃函数。

`hard_reset (bool)` ：是选择硬归零置重置（True）还是减法重置（False）。

### 返回值

`o (torch.Tensor)` ：输出脉冲 $O^{l}$，形状为 `[T, B, ...]`。

`h (torch.Tensor)` ：最终历史电位 $H^{l}(T)$，形状为 `[B, ...]`。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.rand(8, 2, 3) * 2
h = torch.rand(2, 3)
y, h = SF.qif_neuron(x, h, torch.tensor(1.0), torch.tensor(0.0), torch.tensor(2.0), torch.tensor(1.0), torch.tensor(1.0))
print(y)
print(h)
```

## `matterhorn_pytorch.snn.functional.expif_neuron`

ExpIF 神经元的多步函数。

```python
expif_neuron(
    x: torch.Tensor,
    h: torch.Tensor,
    u_threshold: torch.Tensor,
    u_rest: torch.Tensor,
    tau_m: torch.Tensor,
    u_t: torch.Tensor,
    delta_t: torch.Tensor,
    firing: str = "heaviside",
    hard_reset: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]
```

### 参数

`x (torch.Tensor)` ：输入电位 $X^{l}$，形状为 `[T, B, ...]`。

`h (torch.Tensor)` ：初始历史电位 $H^{l}(0)$，形状为 `[B, ...]`。

`u_threshold (torch.Tensor)` ：阈电位。

`u_rest (torch.Tensor)` ：静息电位。

`tau_m (torch.Tensor)` ：神经元时间常数 $\tau_{m}$。

`u_t (torch.Tensor)` ：参数 $u_{T}$。

`delta_t (torch.Tensor)` ：参数 $\Delta_{T}$。

`firing (str)` ：脉冲函数类型，默认为 `heaviside`，即 Heaviside 阶跃函数。

`hard_reset (bool)` ：是选择硬归零置重置（True）还是减法重置（False）。

### 返回值

`o (torch.Tensor)` ：输出脉冲 $O^{l}$，形状为 `[T, B, ...]`。

`h (torch.Tensor)` ：最终历史电位 $H^{l}(T)$，形状为 `[B, ...]`。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.rand(8, 2, 3) * 2
h = torch.rand(2, 3)
y, h = SF.expif_neuron(x, h, torch.tensor(1.0), torch.tensor(0.0), torch.tensor(2.0), torch.tensor(0.0), torch.tensor(0.001))
print(y)
print(h)
```

## `matterhorn_pytorch.snn.functional.izhikevich_neuron`

Izhikevich 神经元的多步函数。

```python
izhikevich_neuron(
    x: torch.Tensor,
    h_w: torch.Tensor,
    u_threshold: torch.Tensor,
    u_rest: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    firing: str = "heaviside",
    hard_reset: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]
```

### 参数

`x (torch.Tensor)` ：输入电位 $X^{l}$，形状为 `[T, B, ...]`。

`h_w (torch.Tensor*)` ：初始历史电位 $H^{l}(0)$ 及状态 $W^{l}(T)$，形状均为 `[B, ...]`。

`u_threshold (torch.Tensor)` ：阈电位。

`u_rest (torch.Tensor)` ：静息电位。

`a (torch.Tensor)` ：参数 $a$。

`b (torch.Tensor)` ：参数 $b$。

`firing (str)` ：脉冲函数类型，默认为 `heaviside`，即 Heaviside 阶跃函数。

`hard_reset (bool)` ：是选择硬归零置重置（True）还是减法重置（False）。

### 返回值

`o (torch.Tensor)` ：输出脉冲 $O^{l}$，形状为 `[T, B, ...]`。

`h_w (torch.Tensor*)` ：最终历史电位 $H^{l}(T)$ 及状态 $W^{l}(T)$，形状均为 `[B, ...]`。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.rand(8, 2, 3) * 2
h = torch.rand(2, 3)
w = torch.zeros(2, 3)
y, (h, w) = SF.izhikevich_neuron(x, (h, w), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0))
print(y)
print(h)
print(w)
```

## `matterhorn_pytorch.snn.functional.klif_neuron`

KLIF 神经元的多步函数。详见参考文献 [2]。

```python
klif_neuron(
    x: torch.Tensor,
    h: torch.Tensor,
    u_threshold: torch.Tensor,
    u_rest: torch.Tensor,
    tau_m: torch.Tensor,
    k: torch.Tensor,
    firing: str = "heaviside",
    hard_reset: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]
```

### 参数

`x (torch.Tensor)` ：输入电位 $X^{l}$，形状为 `[T, B, ...]`。

`h (torch.Tensor)` ：初始历史电位 $H^{l}(0)$，形状为 `[B, ...]`。

`u_threshold (torch.Tensor)` ：阈电位。

`u_rest (torch.Tensor)` ：静息电位。

`tau_m (torch.Tensor)` ：神经元时间常数 $\tau_{m}$。

`k (torch.Tensor)` ：参数 $k$。

`firing (str)` ：脉冲函数类型，默认为 `heaviside`，即 Heaviside 阶跃函数。

`hard_reset (bool)` ：是选择硬归零置重置（True）还是减法重置（False）。

### 返回值

`o (torch.Tensor)` ：输出脉冲 $O^{l}$，形状为 `[T, B, ...]`。

`h (torch.Tensor)` ：最终历史电位 $H^{l}(T)$，形状为 `[B, ...]`。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.rand(8, 2, 3) * 2
h = torch.rand(2, 3)
y, h = SF.klif_neuron(x, h, torch.tensor(1.0), torch.tensor(0.0), torch.tensor(2.0), torch.tensor(2.0))
print(y)
print(h)
```

## `matterhorn_pytorch.snn.functional.lim_neuron`

LIM 神经元的多步函数。与 LIF 神经元不同的是，LIM 神经元输出的为各个时间步的胞体电位。

```python
lim_neuron(
    x: torch.Tensor,
    h: torch.Tensor,
    u_threshold: torch.Tensor,
    u_rest: torch.Tensor,
    tau_m: torch.Tensor,
    firing: str = "heaviside",
    hard_reset: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]
```

### 参数

`x (torch.Tensor)` ：输入电位 $X^{l}$，形状为 `[T, B, ...]`。

`h (torch.Tensor)` ：初始历史电位 $H^{l}(0)$，形状为 `[B, ...]`。

`u_threshold (torch.Tensor)` ：阈电位。

`u_rest (torch.Tensor)` ：静息电位。

`tau_m (torch.Tensor)` ：神经元时间常数 $\tau_{m}$。

`firing (str)` ：脉冲函数类型，默认为 `heaviside`，即 Heaviside 阶跃函数。

`hard_reset (bool)` ：是选择硬归零置重置（True）还是减法重置（False）。

### 返回值

`u (torch.Tensor)` ：输出电位 $U^{l}$，形状为 `[T, B, ...]`。

`h (torch.Tensor)` ：最终历史电位 $H^{l}(T)$，形状为 `[B, ...]`。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.rand(8, 2, 3) * 2
h = torch.rand(2, 3)
y, h = SF.lim_neuron(x, h, torch.tensor(1.0), torch.tensor(0.0), torch.tensor(2.0))
print(y)
print(h)
```

## 参考文献

[1] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.

[2] Jiang C, Zhang Y. Klif: An optimized spiking neuron unit for tuning surrogate gradient slope and membrane potential[J]. arXiv preprint arXiv:2302.09238, 2023.