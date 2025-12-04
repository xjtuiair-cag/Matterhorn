# `matterhorn_pytorch.tnn.functional`

[Back to `matterhorn_pytorch.tnn`](./README.md)

[English](../../en_us/tnn/1_functional.md)

[中文](../../zh_cn/tnn/1_functional.md)

## Module Introduction

This module serves as the function library for the `matterhorn_pytorch.tnn` module, storing the functions that are called. It mainly deals with operators in Spatio-temporal Algebra.

In the following documentation, the time step at which the first spike occurs in a single-neuron spike train is denoted as

$$X_{t}=min(\{t|X(t)=1\})$$

$X_{t}$ represents the time information contained in this spike train. Space-time algebra is an algebraic system used to represent time information and its relationships.

The symbol $\infty$ is used to represent a spike train with no spikes, meaning that the time of the first spike occurrence is infinitely far in the future.

## `matterhorn_pytorch.tnn.functional.t_to_s`

Converts spike times $X_{t}$ to spike train $X$. The formula is as follows:

$$X(t) = (t \ge X_{t}) ? 1 : 0$$

```python
t_to_s(
    t: torch.Tensor,
    time_steps: int,
    t_offset: int = 0
) -> torch.Tensor
```

### Arguments

`t (torch.Tensor)`: Spike times $X_{t}$, with shape `[B, ...]`.

`time_steps (int)`: Time steps `t` after converting to the starting time of the spike train.

`t_offset (int)`: Time offset for converting to the spike train, added before the starting time, with spike times step `T = t_offset + t`.

### Returns

`s (torch.Tensor)`: Spike train $X$, with shape `[T, B, ...]`.

### Example Usage

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
print(x)
y = TF.t_to_s(x, 8)
print(y)
```

## `matterhorn_pytorch.tnn.functional.t_add`

The "delay" operator in space-time algebra. Represented using $+$.

$$Y=X+c \iff Y_{t}=X_{t}+c$$

It represents delaying whole spike train by $c$ time steps.

```python
t_add(
    x: torch.Tensor,
    t: int
) -> torch.Tensor
```

Prefixed with `t_` to indicate it computes spike times, so inputs and outputs are `[B, ...]`.

### Arguments

`x (torch.Tensor)`: Input spike times $X_{t}$.

`t (int)`: Delay time $c$.

### Returns

`y (torch.Tensor)`: Output spike times $Y_{t}$.

### Example Usage

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = torch.randint(6, (10,)).float()
print(x)
y = TF.t_add(x, 3)
print(y)
```

## `matterhorn_pytorch.tnn.functional.t_min`

The "earliest" operator in space-time algebra. Represented using $\veebar$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \veebar B$ | $A$ | $A$ or $B$ | $B$ |

```python
t_min(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `t_` to indicate it computes spike times, so inputs and outputs are `[B, ...]`.

### Arguments

`x (torch.Tensor)`: Input spike times $X_{t}$.

`y (torch.Tensor)`: Input spike times $Y_{t}$.

### Returns

`z (torch.Tensor)`: Output spike times $Z_{t}$.

### Example Usage

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

The "non-simultaneous earliest" operator in space-time algebra. Represented using $\times \veebar$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \times \veebar B$ | $A$ | $\infty$ | $B$ |

```python
t_xmin(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `t_` to indicate it computes spike times, so inputs and outputs are `[B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike times $X_{t}$.

`y (torch.Tensor)`: Input spike times $Y_{t}$.

### Returns

`z (torch.Tensor)`: Output spike times $Z_{t}$.

### Example Usage

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

The "latest" operator in space-time algebra. Represented using $\barwedge$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \barwedge B$ | $B$ | $A$ or $B$ | $A$ |

```python
t_max(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `t_` to indicate it computes spike times, so inputs and outputs are `[B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike times $X_{t}$.

`y (torch.Tensor)`: Input spike times $Y_{t}$.

### Returns

`z (torch.Tensor)`: Output spike times $Z_{t}$.

### Example Usage

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

The "non-simultaneous latest" operator in space-time algebra. Represented using $\times \barwedge$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \times \barwedge B$ | $B$ | $\infty$ | $A$ |

```python
t_xmax(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `t_` to indicate it computes spike times, so inputs and outputs are `[B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike times $X_{t}$.

`y (torch.Tensor)`: Input spike times $Y_{t}$.

### Returns

`z (torch.Tensor)`: Output spike times $Z_{t}$.

### Example Usage

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

The "simultaneous" operator in space-time algebra. Represented using $\equiv$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \equiv B$ | $\infty$ | $A$ | $\infty$ |

```python
t_eq(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `t_` to indicate it computes spike times, so inputs and outputs are `[B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike times $X_{t}$.

`y (torch.Tensor)`: Input spike times $Y_{t}$.

### Returns

`z (torch.Tensor)`: Output spike times $Z_{t}$.

### Example Usage

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

The "non-simultaneous" operator in space-time algebra. Represented using $\ne$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \ne B$ | $A$ | $\infty$ | $A$ |

```python
t_ne(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `t_` to indicate it computes spike times, so inputs and outputs are `[B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike times $X_{t}$.

`y (torch.Tensor)`: Input spike times $Y_{t}$.

### Returns

`z (torch.Tensor)`: Output spike times $Z_{t}$.

### Example Usage

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

The "before" operator in space-time algebra. Represented using $\prec$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \prec B$ | $A$ | $\infty$ | $\infty$ |

```python
t_lt(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `t_` to indicate it computes spike times, so inputs and outputs are `[B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike times $X_{t}$.

`y (torch.Tensor)`: Input spike times $Y_{t}$.

### Returns

`z (torch.Tensor)`: Output spike times $Z_{t}$.

### Example Usage

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

The "before or simultaneous" operator in space-time algebra. Represented using $\preccurlyeq$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \preccurlyeq B$ | $A$ | $A$ | $\infty$ |

```python
t_le(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `t_` to indicate it computes spike times, so inputs and outputs are `[B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike times $X_{t}$.

`y (torch.Tensor)`: Input spike times $Y_{t}$.

### Returns

`z (torch.Tensor)`: Output spike times $Z_{t}$.

### Example Usage

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

The "after" operator in space-time algebra. Represented using $\succ$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \succ B$ | $\infty$ | $\infty$ | $A$ |

```python
t_gt(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `t_` to indicate it computes spike times, so inputs and outputs are `[B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike times $X_{t}$.

`y (torch.Tensor)`: Input spike times $Y_{t}$.

### Returns

`z (torch.Tensor)`: Output spike times $Z_{t}$.

### Example Usage

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

The "after or simultaneous" operator in space-time algebra. Represented using $\succcurlyeq$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \succcurlyeq B$ | $\infty$ | $A$ | $A$ |

```python
t_ge(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `t_` to indicate it computes spike times, so inputs and outputs are `[B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike times $X_{t}$.

`y (torch.Tensor)`: Input spike times $Y_{t}$.

### Returns

`z (torch.Tensor)`: Output spike times $Z_{t}$.

### Example Usage

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

Converts a spike train $X$ to spike times $X_{t}$. The formula is as follows:

$$X_{t}=min(\{t|X(t)=1\})$$

Returns $\infty$ if there are no spikes in the sequence.

```python
s_to_t(
    s: torch.Tensor
) -> torch.Tensor
```

### Parameters

`s (torch.Tensor)`: Spike train $X$, shape `[T, B, ...]`.

### Returns

`t (torch.Tensor)`: Spike times $X_{t}$, shape `[B, ...]`.

### Example Usage

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

The "delay" operator in space-time algebra. Represented using $+$.

$$Y=X+c \iff Y_{t}=X_{t}+c$$

Represents delaying whole spike train by $c$ time steps.

```python
s_add(
    x: torch.Tensor,
    t: int
) -> torch.Tensor
```

Prefixed with `s_` to indicate it computes spike train, so inputs and outputs are `[T, B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike train $X$.

`t (int)`: Delay time $c$.

### Returns

`y (torch.Tensor)`: Output signal $Y$.

### Example Usage

```python
import torch
import matterhorn_pytorch.tnn.functional as TF


x = TF.t_to_s(torch.randint(6, (10,)).float(), 8)
print(x)
y = TF.s_add(x, 3)
print(y)
```

## `matterhorn_pytorch.tnn.functional.s_min`

The "earliest" operator in space-time algebra. Represented using $\veebar$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \veebar B$ | $A$ | $A$ or $B$ | $B$ |

```python
s_min(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `s_` to indicate it computes spike train, so inputs and outputs are `[T, B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike train $X$.

`y (torch.Tensor)`: Input spike train $Y$.

### Returns

`z (torch.Tensor)`: Output spike train $Z$.

### Example Usage

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

The "non-simultaneous earliest" operator in space-time algebra. Represented using $\times \veebar$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \times \veebar B$ | $A$ | $\infty$ | $B$ |

```python
s_xmin(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `s_` to indicate it computes spike train, so inputs and outputs are `[T, B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike train $X$.

`y (torch.Tensor)`: Input spike train $Y$.

### Returns

`z (torch.Tensor)`: Output spike train $Z$.

### Example Usage

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

The "latest" operator in space-time algebra. Represented using $\barwedge$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \barwedge B$ | $B$ | $A$ or $B$ | $A$ |

```python
s_max(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `s_` to indicate it computes spike train, so inputs and outputs are `[T, B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike train $X$.

`y (torch.Tensor)`: Input spike train $Y$.

### Returns

`z (torch.Tensor)`: Output spike train $Z$.

### Example Usage

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

The "non-simultaneous latest" operator in space-time algebra. Represented using $\times \barwedge$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \times \barwedge B$ | $B$ | $\infty$ | $A$ |

```python
s_xmax(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `s_` to indicate it computes spike train, so inputs and outputs are `[T, B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike train $X$.

`y (torch.Tensor)`: Input spike train $Y$.

### Returns

`z (torch.Tensor)`: Output spike train $Z$.

### Example Usage

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

The "simultaneous" operator in space-time algebra. Represented using $\equiv$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \equiv B$ | $\infty$ | $A$ | $\infty$ |

```python
s_eq(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `s_` to indicate it computes spike train, so inputs and outputs are `[T, B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike train $X$.

`y (torch.Tensor)`: Input spike train $Y$.

### Returns

`z (torch.Tensor)`: Output spike train $Z$.

### Example Usage

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

The "non-simultaneous" operator in space-time algebra. Represented using $\ne$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \ne B$ | $A$ | $\infty$ | $A$ |

```python
s_ne(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `s_` to indicate it computes spike train, so inputs and outputs are `[T, B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike train $X$.

`y (torch.Tensor)`: Input spike train $Y$.

### Returns

`z (torch.Tensor)`: Output spike train $Z$.

### Example Usage

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

The "before" operator in space-time algebra. Represented using $\prec$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \prec B$ | $A$ | $\infty$ | $\infty$ |

```python
s_lt(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `s_` to indicate it computes spike train, so inputs and outputs are `[T, B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike train $X$.

`y (torch.Tensor)`: Input spike train $Y$.

### Returns

`z (torch.Tensor)`: Output spike train $Z$.

### Example Usage

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

The "before or simultaneous" operator in space-time algebra. Represented using $\preccurlyeq$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \preccurlyeq B$ | $A$ | $A$ | $\infty$ |

```python
s_le(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `s_` to indicate it computes spike train, so inputs and outputs are `[T, B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike train $X$.

`y (torch.Tensor)`: Input spike train $Y$.

### Returns

`z (torch.Tensor)`: Output spike train $Z$.

### Example Usage

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

The "after" operator in space-time algebra. Represented using $\succ$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \succ B$ | $\infty$ | $\infty$ | $A$ |

```python
s_gt(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `s_` to indicate it computes spike train, so inputs and outputs are `[T, B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike train $X$.

`y (torch.Tensor)`: Input spike train $Y$.

### Returns

`z (torch.Tensor)`: Output spike train $Z$.

### Example Usage

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

The "after or simultaneous" operator in space-time algebra. Represented using $\succcurlyeq$.

| Operator | $A_{t} < B_{t}$ | $A_{t} = B_{t}$ | $A_{t} > B_{t}$ |
| :---: | :---: | :---: | :---: |
| $A \succcurlyeq B$ | $\infty$ | $A$ | $A$ |

```python
s_ge(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Prefixed with `s_` to indicate it computes spike train, so inputs and outputs are `[T, B, ...]`.

### Parameters

`x (torch.Tensor)`: Input spike train $X$.

`y (torch.Tensor)`: Input spike train $Y$.

### Returns

`z (torch.Tensor)`: Output spike train $Z$.

### Example Usage

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

## References

[1] Hestenes D. Space-time algebra[M]. Switzerland: Springer International Publishing, 2015.

[2] Smith J. Space-time algebra: A model for neocortical computation[C]//2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA). IEEE, 2018: 289-300.