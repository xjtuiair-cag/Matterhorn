# `matterhorn_pytorch.snn.functional`

[Back to `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/1_functional.md)

[中文](../../zh_cn/snn/1_functional.md)

## Module Introduction

This module is a function library for the `matterhorn_pytorch.snn` module, storing functions that are called.

## `matterhorn_pytorch.snn.functional.val_to_spike`

Converts values to spikes (either on or off) with $x \ge 0.5$ as the threshold.

```python
val_to_spike(
    x: torch.Tensor
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$, analog value.

### Returns

`o (torch.Tensor)`: Output $O$, spike train.

### Example Usage

```python
import torch
import matterhorn.snn.functional as SF


x = torch.rand(2, 3)
print(x)
y = SF.val_to_spike(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.heaviside_rectangular`

Heaviside step function:

$$u(x)=x \ge 0 ? 1 : 0$$

With a rectangular window as the gradient for backpropagation. Detailed definitions can be found in reference [1].

```python
heaviside_rectangular(
    x: torch.Tensor,
    a: float = 1.0
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$, membrane voltage.

`a (float)`: Parameter $a$.

### Returns

`o (torch.Tensor)`: Output $O$, spike train.

### Example Usage

```python
import torch
import matterhorn.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.heaviside_rectangular(x, 1.0)
print(y)
```

## `matterhorn_pytorch.snn.functional.heaviside_polynomial`

Heaviside step function:

$$u(x)=x \ge 0 ? 1 : 0$$

With a triangular window as the gradient for backpropagation. Detailed definitions can be found in reference [1].

```python
heaviside_polynomial(
    x: torch.Tensor,
    a: float = 4.0
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$, membrane voltage.

`a (float)`: Parameter $a$.

### Returns

`o (torch.Tensor)`: Output $O$, spike train.

### Example Usage

```python
import torch
import matterhorn.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.heaviside_polynomial(x, 4.0)
print(y)
```

## `matterhorn_pytorch.snn.functional.heaviside_sigmoid`

Heaviside step function:

$$u(x)=x \ge 0 ? 1 : 0$$

With the gradient of the Sigmoid function for backpropagation. Detailed definitions can be found in reference [1].

```python
heaviside_sigmoid(
    x: torch.Tensor,
    a: float = 0.25
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$, membrane voltage.

`a (float)`: Parameter $a$.

### Returns

`o (torch.Tensor)`: Output $O$, spike train.

### Example Usage

```python
import torch
import matterhorn.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.heaviside_sigmoid(x, 0.25)
print(y)
```

## `matterhorn_pytorch.snn.functional.heaviside_gaussian`

Heaviside step function:

$$u(x)=x \ge 0 ? 1 : 0$$

With a Gaussian function as the gradient for backpropagation. Detailed definitions can be found in reference [1].

```python
heaviside_gaussian(
    x: torch.Tensor,
    a: float = 0.16
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$, membrane voltage.

`a (float)`: Parameter $a$.

### Returns

`o (torch.Tensor)`: Output $O$, spike train.

### Example Usage

```python
import torch
import matterhorn.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.heaviside_gaussian(x, 0.16)
print(y)
```

## References

[1] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.