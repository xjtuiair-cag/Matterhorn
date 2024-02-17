# `matterhorn_pytorch.tnn.functional`

[回到 `matterhorn_pytorch.tnn`](./README.md)

[English](../../en_us/tnn/1_functional.md)

[中文](../../zh_cn/tnn/1_functional.md)

## 模块简介

该模块为 `matterhorn_pytorch.tnn` 模块的函数库，存储所被调用的函数。主要为时空代数（Space-time Algebra）中的运算符。

在以下文档中，将单个神经元脉冲序列中首个脉冲产生的时间步记作

$$X_{t}=min(\{t|X(t)=1\})$$

$X_{t}$ 代表这段脉冲序列中所含有的时间信息。时空代数则是用来表示时间信息及其之间关系的代数体系。

使用 $\infty$ 代表没有任何脉冲的脉冲序列，其物理意义为：首个脉冲产生的时间为无穷远之后。

## `matterhorn_pytorch.tnn.functional.t_to_s`

将脉冲时间 $X_{t}$ 转为脉冲序列 $X$ 。公式如下：

$$X(t) = (t \ge X_{t}) ? 1 : 0$$

```python
t_to_s(
    t: torch.Tensor,
    time_steps: int,
    t_offset: int = 0
) -> torch.Tensor
```

### 参数

`t (torch.Tensor)` ：脉冲时间 $X_{t}$ ，形状为 `[B, ...]` 。

`time_steps (int)` ：转换成脉冲序列起始时间后的时间步长 `t` 。

`t_offset (int)` ：转换成脉冲序列的时间偏移量，加在起始时间前，脉冲时间步长为 `T = t_offset + t` 。

### 返回值

`s (torch.Tensor)` ：脉冲序列 $X$ ，形状为 `[T, B, ...]` 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
print(x)
y = TF.t_to_s(x, 8)
print(y)
```

## `matterhorn_pytorch.tnn.functional.t_add`

时空代数的“延迟”算子。使用 $+$ 表示。

$$Y=X+c \iff Y_{t}=X_{t}+c$$

代表脉冲整体向后推移 $c$ 个时间步。

```python
t_add(
    x: torch.Tensor,
    t: int
) -> torch.Tensor
```

以 `t_` 开头表明其为计算脉冲时间的函数，因此输入和输出为 `[B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲时间 $X_{t}$ 。

`t (int)` ：延迟时间 $c$ 。

### 返回值

`y (torch.Tensor)` ：输出脉冲时间 $Y_{t}$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
print(x)
y = TF.t_add(x, 3)
print(y)
```

## `matterhorn_pytorch.tnn.functional.t_min`

时空代数的“最早”算子。使用 $\veebar$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \veebar B$ | $A$ | $A$ 或 $B$ | $B$ |

```python
t_min(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `t_` 开头表明其为计算脉冲时间的函数，因此输入和输出为 `[B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲时间 $X_{t}$ 。

`y (torch.Tensor)` ：输入脉冲时间 $Y_{t}$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲时间 $Z_{t}$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
y = torch.randint(6, (10,)).float()
print(x)
print(y)
z = TF.t_min(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.t_xmin`

时空代数的“非同时最早”算子。使用 $\times \veebar$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \times \veebar B$ | $A$ | $\infty$ | $B$ |

```python
t_xmin(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `t_` 开头表明其为计算脉冲时间的函数，因此输入和输出为 `[B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲时间 $X_{t}$ 。

`y (torch.Tensor)` ：输入脉冲时间 $Y_{t}$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲时间 $Z_{t}$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
y = torch.randint(6, (10,)).float()
print(x)
print(y)
z = TF.t_xmin(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.t_max`

时空代数的“最晚”算子。使用 $\barwedge$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \barwedge B$ | $B$ | $A$ 或 $B$ | $A$ |

```python
t_max(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `t_` 开头表明其为计算脉冲时间的函数，因此输入和输出为 `[B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲时间 $X_{t}$ 。

`y (torch.Tensor)` ：输入脉冲时间 $Y_{t}$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲时间 $Z_{t}$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
y = torch.randint(6, (10,)).float()
print(x)
print(y)
z = TF.t_max(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.t_xmax`

时空代数的“非同时最晚”算子。使用 $\times \barwedge$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \times \barwedge B$ | $B$ | $\infty$ | $A$ |

```python
t_xmax(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `t_` 开头表明其为计算脉冲时间的函数，因此输入和输出为 `[B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲时间 $X_{t}$ 。

`y (torch.Tensor)` ：输入脉冲时间 $Y_{t}$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲时间 $Z_{t}$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
y = torch.randint(6, (10,)).float()
print(x)
print(y)
z = TF.t_xmax(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.t_eq`

时空代数的“同时”算子。使用 $\equiv$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \equiv B$ | $\infty$ | $A$ | $\infty$ |

```python
t_eq(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `t_` 开头表明其为计算脉冲时间的函数，因此输入和输出为 `[B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲时间 $X_{t}$ 。

`y (torch.Tensor)` ：输入脉冲时间 $Y_{t}$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲时间 $Z_{t}$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
y = torch.randint(6, (10,)).float()
print(x)
print(y)
z = TF.t_eq(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.t_ne`

时空代数的“不同时”算子。使用 $\ne$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \ne B$ | $A$ | $\infty$ | $A$ |

```python
t_ne(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `t_` 开头表明其为计算脉冲时间的函数，因此输入和输出为 `[B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲时间 $X_{t}$ 。

`y (torch.Tensor)` ：输入脉冲时间 $Y_{t}$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲时间 $Z_{t}$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
y = torch.randint(6, (10,)).float()
print(x)
print(y)
z = TF.t_ne(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.t_lt`

时空代数的“早于”算子。使用 $\prec$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \prec B$ | $A$ | $\infty$ | $\infty$ |

```python
t_lt(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `t_` 开头表明其为计算脉冲时间的函数，因此输入和输出为 `[B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲时间 $X_{t}$ 。

`y (torch.Tensor)` ：输入脉冲时间 $Y_{t}$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲时间 $Z_{t}$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
y = torch.randint(6, (10,)).float()
print(x)
print(y)
z = TF.t_lt(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.t_le`

时空代数的“早于或同时”算子。使用 $\preccurlyeq$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \preccurlyeq B$ | $A$ | $A$ | $\infty$ |

```python
t_le(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `t_` 开头表明其为计算脉冲时间的函数，因此输入和输出为 `[B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲时间 $X_{t}$ 。

`y (torch.Tensor)` ：输入脉冲时间 $Y_{t}$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲时间 $Z_{t}$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
y = torch.randint(6, (10,)).float()
print(x)
print(y)
z = TF.t_le(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.t_gt`

时空代数的“晚于”算子。使用 $\succ$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \succ B$ | $\infty$ | $\infty$ | $A$ |

```python
t_gt(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `t_` 开头表明其为计算脉冲时间的函数，因此输入和输出为 `[B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲时间 $X_{t}$ 。

`y (torch.Tensor)` ：输入脉冲时间 $Y_{t}$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲时间 $Z_{t}$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
y = torch.randint(6, (10,)).float()
print(x)
print(y)
z = TF.t_gt(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.t_ge`

时空代数的“晚于或同时”算子。使用 $\succcurlyeq$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \succcurlyeq B$ | $\infty$ | $A$ | $A$ |

```python
t_ge(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `t_` 开头表明其为计算脉冲时间的函数，因此输入和输出为 `[B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲时间 $X_{t}$ 。

`y (torch.Tensor)` ：输入脉冲时间 $Y_{t}$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲时间 $Z_{t}$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
y = torch.randint(6, (10,)).float()
print(x)
print(y)
z = TF.t_ge(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.s_to_t`

将脉冲序列 $X$ 转为脉冲时间 $X_{t}$ 。公式如下：

$$X_{t}=min(\{t|X(t)=1\})$$

若脉冲序列上无脉冲，则输出 $∞$ 。

```python
s_to_t(
    s: torch.Tensor
) -> torch.Tensor
```

### 参数

`s (torch.Tensor)` ：脉冲序列 $X$ ，形状为 `[T, B, ...]` 。

### 返回值

`t (torch.Tensor)` ：脉冲时间 $X_{t}$ ，形状为 `[B, ...]` 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0]
], dtype = torch.float)
print(x)
y = TF.s_to_t(x)
print(y)
```

## `matterhorn_pytorch.tnn.functional.s_add`

时空代数的“延迟”算子。使用 $+$ 表示。

$$Y=X+c \iff Y_{t}=X_{t}+c$$

代表脉冲整体向后推移 $c$ 个时间步。

```python
s_add(
    x: torch.Tensor,
    t: int
) -> torch.Tensor
```

以 `s_` 开头表明其为计算脉冲序列的函数，因此输入和输出为 `[T, B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲序列 $X$ 。

`t (int)` ：延迟时间 $c$ 。

### 返回值

`y (torch.Tensor)` ：输出信号 $Y$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
print(x)
y = TF.s_add(x, 3)
print(y)
```

## `matterhorn_pytorch.tnn.functional.s_min`

时空代数的“最早”算子。使用 $\veebar$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \veebar B$ | $A$ | $A$ 或 $B$ | $B$ |

```python
s_min(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `s_` 开头表明其为计算脉冲序列的函数，因此输入和输出为 `[T, B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲序列 $X$ 。

`y (torch.Tensor)` ：输入脉冲序列 $Y$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲序列 $Z$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
y = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
print(x)
print(y)
z = TF.s_min(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.s_xmin`

时空代数的“非同时最早”算子。使用 $\times \veebar$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \times \veebar B$ | $A$ | $\infty$ | $B$ |

```python
s_xmin(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `s_` 开头表明其为计算脉冲序列的函数，因此输入和输出为 `[T, B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲序列 $X$ 。

`y (torch.Tensor)` ：输入脉冲序列 $Y$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲序列 $Z$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
y = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
print(x)
print(y)
z = TF.s_xmin(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.s_max`

时空代数的“最晚”算子。使用 $\barwedge$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \barwedge B$ | $B$ | $A$ 或 $B$ | $A$ |

```python
s_max(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `s_` 开头表明其为计算脉冲序列的函数，因此输入和输出为 `[T, B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲序列 $X$ 。

`y (torch.Tensor)` ：输入脉冲序列 $Y$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲序列 $Z$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
y = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
print(x)
print(y)
z = TF.s_max(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.s_xmax`

时空代数的“非同时最晚”算子。使用 $\times \barwedge$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \times \barwedge B$ | $B$ | $\infty$ | $A$ |

```python
s_xmax(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `s_` 开头表明其为计算脉冲序列的函数，因此输入和输出为 `[T, B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲序列 $X$ 。

`y (torch.Tensor)` ：输入脉冲序列 $Y$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲序列 $Z$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
y = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
print(x)
print(y)
z = TF.s_xmax(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.s_eq`

时空代数的“同时”算子。使用 $\equiv$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \equiv B$ | $\infty$ | $A$ | $\infty$ |

```python
s_eq(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `s_` 开头表明其为计算脉冲序列的函数，因此输入和输出为 `[T, B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲序列 $X$ 。

`y (torch.Tensor)` ：输入脉冲序列 $Y$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲序列 $Z$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
y = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
print(x)
print(y)
z = TF.s_eq(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.s_ne`

时空代数的“不同时”算子。使用 $\ne$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \ne B$ | $A$ | $\infty$ | $A$ |

```python
s_ne(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `s_` 开头表明其为计算脉冲序列的函数，因此输入和输出为 `[T, B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲序列 $X$ 。

`y (torch.Tensor)` ：输入脉冲序列 $Y$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲序列 $Z$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
y = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
print(x)
print(y)
z = TF.s_ne(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.s_lt`

时空代数的“早于”算子。使用 $\prec$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \prec B$ | $A$ | $\infty$ | $\infty$ |

```python
s_lt(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `s_` 开头表明其为计算脉冲序列的函数，因此输入和输出为 `[T, B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲序列 $X$ 。

`y (torch.Tensor)` ：输入脉冲序列 $Y$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲序列 $Z$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
y = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
print(x)
print(y)
z = TF.s_lt(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.s_le`

时空代数的“早于或同时”算子。使用 $\preccurlyeq$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \preccurlyeq B$ | $A$ | $A$ | $\infty$ |

```python
s_le(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `s_` 开头表明其为计算脉冲序列的函数，因此输入和输出为 `[T, B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲序列 $X$ 。

`y (torch.Tensor)` ：输入脉冲序列 $Y$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲序列 $Z$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
y = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
print(x)
print(y)
z = TF.s_le(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.s_gt`

时空代数的“晚于”算子。使用 $\succ$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \succ B$ | $\infty$ | $\infty$ | $A$ |

```python
s_gt(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `s_` 开头表明其为计算脉冲序列的函数，因此输入和输出为 `[T, B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲序列 $X$ 。

`y (torch.Tensor)` ：输入脉冲序列 $Y$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲序列 $Z$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
y = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
print(x)
print(y)
z = TF.s_gt(x, y)
print(z)
```

## `matterhorn_pytorch.tnn.functional.s_ge`

时空代数的“晚于或同时”算子。使用 $\succcurlyeq$ 表示。

| 算子 | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \succcurlyeq B$ | $\infty$ | $A$ | $A$ |

```python
s_ge(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

以 `s_` 开头表明其为计算脉冲序列的函数，因此输入和输出为 `[T, B, ...]` 。

### 参数

`x (torch.Tensor)` ：输入脉冲序列 $X$ 。

`y (torch.Tensor)` ：输入脉冲序列 $Y$ 。

### 返回值

`z (torch.Tensor)` ：输出脉冲序列 $Z$ 。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
y = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
print(x)
print(y)
z = TF.s_ge(x, y)
print(z)
```

## 参考文献

[1] Hestenes D. Space-time algebra[M]. Switzerland: Springer International Publishing, 2015.

[2] Smith J. Space-time algebra: A model for neocortical computation[C]//2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA). IEEE, 2018: 289-300.