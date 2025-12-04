# `matterhorn_pytorch.snn.functional`

[Back to `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/1_functional.md)

[中文](../../zh_cn/snn/1_functional.md)

## Module Introduction

This module is a function library for the `matterhorn_pytorch.snn` module, storing functions that are called.

## `matterhorn_pytorch.snn.functional.to_spike_train`

Converts values to spikes (either on or off) with $x \ge 0.5$ as the threshold.

```python
to_spike_train(
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
import matterhorn_pytorch.snn.functional as SF


x = torch.rand(2, 3)
print(x)
y = SF.to_spike_train(x)
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
import matterhorn_pytorch.snn.functional as SF


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
import matterhorn_pytorch.snn.functional as SF


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
import matterhorn_pytorch.snn.functional as SF


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
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.heaviside_gaussian(x, 0.16)
print(y)
```

## `matterhorn_pytorch.snn.functional.lt`

The "less than" operator implemented based on the step function $u(t)$, using a Gaussian function as the gradient for backpropagation.

```python
lt(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$.

`y (torch.Tensor)`: Input $Y$.

### Returns

`z (torch.Tensor)`: Output $Z:=(X<Y)?1:0$.

### Example Usage

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

The "less than or equal to" operator implemented based on the step function $u(t)$, using a Gaussian function as the gradient for backpropagation.

```python
le(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$.

`y (torch.Tensor)`: Input $Y$.

### Returns

`z (torch.Tensor)`: Output $Z:=(X \le Y)?1:0$.

### Example Usage

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

The "greater than" operator implemented based on the step function $u(t)$, using a Gaussian function as the gradient for backpropagation.

```python
gt(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$.

`y (torch.Tensor)`: Input $Y$.

### Returns

`z (torch.Tensor)`: Output $Z:=(X>Y)?1:0$.

### Example Usage

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

The "greater than or equal to" operator implemented based on the step function $u(t)$, using a Gaussian function as the gradient for backpropagation.

```python
ge(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$.

`y (torch.Tensor)`: Input $Y$.

### Returns

`z (torch.Tensor)`: Output $Z:=(X \ge Y)?1:0$.

### Example Usage

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

The "equal to" operator implemented based on the step function $u(t)$, using a Gaussian function as the gradient for backpropagation.

```python
eq(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$.

`y (torch.Tensor)`: Input $Y$.

### Returns

`z (torch.Tensor)`: Output $Z:=(X=Y)?1:0$.

### Example Usage

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

The "not equal to" operator implemented based on the step function $u(t)$, using a Gaussian function as the gradient for backpropagation.

```python
ne(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$.

`y (torch.Tensor)`: Input $Y$.

### Returns

`z (torch.Tensor)`: Output $Z:=(X \neq Y)?1:0$.

### Example Usage

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

The "floor" operator implemented based on a multi-step step function.

```python
floor(
    x: torch.Tensor
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$.

### Returns

`y (torch.Tensor)`: Output $Y=\lfloor X \rfloor$.

### Example Usage

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.floor(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.ceil`

The "ceil" operator implemented based on a multi-step step function.

```python
ceil(
    x: torch.Tensor
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$.

### Returns

`y (torch.Tensor)`: Output $Y=\lceil X \rceil$.

### Example Usage

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.ceil(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.round`

The "round" operator implemented based on a multi-step step function.

```python
round(
    x: torch.Tensor
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Input $X$.

### Returns

`y (torch.Tensor)`: Output $Y=\lfloor X + 0.5 \rfloor$.

### Example Usage

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(-2, 3, size = (2, 3))
print(x)
y = SF.round(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.encode_poisson`

Poisson encoding of spikes, assuming the number of spikes generated at each time step follows a Poisson distribution. A type of rate coding.

```python
encode_poisson(
    x: torch.Tensor,
    precision: float = 1e-5,
    count: bool = True
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Analog value $X$, serving as the spike firing probability.

`precision (float)`: Since $\lambda > 0$ in a Poisson distribution, this parameter specifies the lower bound for $\lambda$.

`count (bool)`: Whether to send the spike count.

### Returns

`y (torch.Tensor)`: Output spikes $O^{0}$.

### Example Usage

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.rand(2, 3)
print(x)
y = SF.encode_poisson(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.encode_temporal`

Temporal encoding of spikes. When the current time step exceeds the incoming time step value, a spike is emitted with probability `prob`.

```python
encode_temporal(
    x: torch.Tensor,
    time_steps: int,
    t_offset: int = 0,
    prob: float = 1.0
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Analog value $X$, serving as the time step to start firing spikes.

`time_steps (int)`: Total number of time steps to generate.

`t_offset (int)`: Offset for the current time step, i.e., from which time step to start counting.

`prob (float)`: Probability of sending a spike after reaching the designated start firing time step.

### Returns

`y (torch.Tensor)`: Output spikes $O^{0}$.

### Example Usage

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(0, 8, (2, 3))
print(x)
y = SF.encode_temporal(x, 8)
print(y)
```

## `matterhorn_pytorch.snn.functional.encode_binary`

Binary phase encoding of spikes. Encodes an integer into a spike sequence in binary form, following big-endian order (higher bits are encoded earlier).

```python
encode_binary(
    x: torch.Tensor,
    length: int = 8,
    repeat: int = 1
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Analog value $X$, an integer between $[0,2^{T}-1]$.

`length (int)`: Time step length $T$, i.e., bit width. For example, `length = 8` represents integers between $[0,255]$.

`repeat (int)`: Number of repetitions.

### Returns

`y (torch.Tensor)`: Output spikes $O^{0}$.

### Example Usage

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(0, 256, (2, 3))
print(x)
y = SF.encode_binary(x, 8)
print(y)
```

## `matterhorn_pytorch.snn.functional.decode_sum_spike`

Spike decoding implemented by calculating the total sum of spikes.

```python
decode_sum_spike(
    x: torch.Tensor
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Output spikes $O^{L}$.

### Returns

`y (torch.Tensor)`: Total spike sum $Y=\sum_{t=0}^{T-1}O^{L}(t)$.

### Example Usage

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(0, 2, size = (8, 2, 3)).float()
print(x)
y = SF.decode_sum_spike(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.decode_avg_spike`

Spike decoding implemented by calculating the average of spikes.

```python
decode_avg_spike(
    x: torch.Tensor
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Output spikes $O^{L}$.

### Returns

`y (torch.Tensor)`: Average spikes $Y=\frac{\sum_{t=0}^{T-1}O^{L}(t)}{T}$.

### Example Usage

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(0, 2, size = (8, 2, 3)).float()
print(x)
y = SF.decode_avg_spike(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.decode_min_time`

Spike decoding implemented by calculating the earliest occurrence time of spikes.

```python
decode_min_time(
    x: torch.Tensor,
    t_offset: int = 0,
    empty_fill: float = -1
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Output spikes $O^{L}$.

`t_offset (int)`: Time step offset $t_{0}$.

`empty_fill (float)`: Value to fill with if the spike train contains no spikes (all zeros). Defaults to $-1$.

### Returns

`y (torch.Tensor)`: The earliest time step containing a spike $Y=min\{t_{i}|O^{l}(t_{i})=1\}$.

### Example Usage

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(0, 2, size = (8, 2, 3)).float()
print(x)
y = SF.decode_min_time(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.decode_avg_time`

Spike decoding implemented by calculating the average occurrence time of spikes.

```python
decode_avg_time(
    x: torch.Tensor,
    t_offset: int = 0,
    empty_fill: float = -1
) -> torch.Tensor
```

### Parameters

`x (torch.Tensor)`: Output spikes $O^{L}$.

`t_offset (int)`: Time step offset $t_{0}$.

`empty_fill (float)`: Value to fill with if the spike train contains no spikes (all zeros). Defaults to $-1$.

### Returns

`y (torch.Tensor)`: The average time step of spikes $Y=\frac{\sum_{i}{\{t_{i}|O^{l}(t_{i})=1\}}}{n}$, where $n=\sum_{i}{\{1|O^{l}(t_{i})=1\}}$ is the number of spikes.

### Example Usage

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.randint(0, 2, size = (8, 2, 3)).float()
print(x)
y = SF.decode_avg_time(x)
print(y)
```

## `matterhorn_pytorch.snn.functional.if_neuron`

Multi-step function for the IF neuron.

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

### Parameters

`x (torch.Tensor)`: Input potential $X^{l}$ with shape `[T, B, ...]`.

`h (torch.Tensor)`: Initial historical potential $H^{l}(0)$ with shape `[B, ...]`.

`u_threshold (torch.Tensor)`: Threshold potential.

`u_rest (torch.Tensor)`: Resting potential.

`firing (str)`: Type of spiking function, defaults to `heaviside`, i.e., the Heaviside step function.

`hard_reset (bool)`: Whether to choose hard reset to zero (True) or subtractive reset (False).

### Returns

`o (torch.Tensor)`: Output spikes $O^{l}$ with shape `[T, B, ...]`.

`h (torch.Tensor)`: Final historical potential $H^{l}(T)$ with shape `[B, ...]`.

### Example Usage

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

Multi-step function for the LIF neuron.

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

### Parameters

`x (torch.Tensor)`: Input potential $X^{l}$ with shape `[T, B, ...]`.

`h (torch.Tensor)`: Initial historical potential $H^{l}(0)$ with shape `[B, ...]`.

`u_threshold (torch.Tensor)`: Threshold potential.

`u_rest (torch.Tensor)`: Resting potential.

`tau_m (torch.Tensor)`: Neuron time constant $\tau_{m}$.

`firing (str)`: Type of spiking function, defaults to `heaviside`, i.e., the Heaviside step function.

`hard_reset (bool)`: Whether to choose hard reset to zero (True) or subtractive reset (False).

### Returns

`o (torch.Tensor)`: Output spikes $O^{l}$ with shape `[T, B, ...]`.

`h (torch.Tensor)`: Final historical potential $H^{l}(T)$ with shape `[B, ...]`.

### Example Usage

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

Multi-step function for the QIF neuron.

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

### Parameters

`x (torch.Tensor)`: Input potential $X^{l}$ with shape `[T, B, ...]`.

`h (torch.Tensor)`: Initial historical potential $H^{l}(0)$ with shape `[B, ...]`.

`u_threshold (torch.Tensor)`: Threshold potential.

`u_rest (torch.Tensor)`: Resting potential.

`tau_m (torch.Tensor)`: Neuron time constant $\tau_{m}$.

`u_c (torch.Tensor)`: Parameter $u_{c}$.

`a_0 (torch.Tensor)`: Parameter $a_{0}$.

`firing (str)`: Type of spiking function, defaults to `heaviside`, i.e., the Heaviside step function.

`hard_reset (bool)`: Whether to choose hard reset to zero (True) or subtractive reset (False).

### Returns

`o (torch.Tensor)`: Output spikes $O^{l}$ with shape `[T, B, ...]`.

`h (torch.Tensor)`: Final historical potential $H^{l}(T)$ with shape `[B, ...]`.

### Example Usage

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

Multi-step function for the ExpIF neuron.

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

### Parameters

`x (torch.Tensor)`: Input potential $X^{l}$ with shape `[T, B, ...]`.

`h (torch.Tensor)`: Initial historical potential $H^{l}(0)$ with shape `[B, ...]`.

`u_threshold (torch.Tensor)`: Threshold potential.

`u_rest (torch.Tensor)`: Resting potential.

`tau_m (torch.Tensor)`: Neuron time constant $\tau_{m}$.

`u_t (torch.Tensor)`: Parameter $u_{T}$.

`delta_t (torch.Tensor)`: Parameter $\Delta_{T}$.

`firing (str)`: Type of spiking function, defaults to `heaviside`, i.e., the Heaviside step function.

`hard_reset (bool)`: Whether to choose hard reset to zero (True) or subtractive reset (False).

### Returns

`o (torch.Tensor)`: Output spikes $O^{l}$ with shape `[T, B, ...]`.

`h (torch.Tensor)`: Final historical potential $H^{l}(T)$ with shape `[B, ...]`.

### Example Usage

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

Multi-step function for the Izhikevich neuron.

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

### Parameters

`x (torch.Tensor)`: Input potential $X^{l}$ with shape `[T, B, ...]`.

`h_w (torch.Tensor*)`: Initial historical potential $H^{l}(0)$ and state $W^{l}(T)$, both with shape `[B, ...]`.

`u_threshold (torch.Tensor)`: Threshold potential.

`u_rest (torch.Tensor)`: Resting potential.

`a (torch.Tensor)`: Parameter $a$.

`b (torch.Tensor)`: Parameter $b$.

`firing (str)`: Type of spiking function, defaults to `heaviside`, i.e., the Heaviside step function.

`hard_reset (bool)`: Whether to choose hard reset to zero (True) or subtractive reset (False).

### Returns

`o (torch.Tensor)`: Output spikes $O^{l}$ with shape `[T, B, ...]`.

`h_w (torch.Tensor*)`: Final historical potential $H^{l}(T)$ and state $W^{l}(T)$, both with shape `[B, ...]`.

### Example Usage

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

Multi-step function for the KLIF neuron. For details, see reference [2].

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

### Parameters

`x (torch.Tensor)`: Input potential $X^{l}$ with shape `[T, B, ...]`.

`h (torch.Tensor)`: Initial historical potential $H^{l}(0)$ with shape `[B, ...]`.

`u_threshold (torch.Tensor)`: Threshold potential.

`u_rest (torch.Tensor)`: Resting potential.

`tau_m (torch.Tensor)`: Neuron time constant $\tau_{m}$.

`k (torch.Tensor)`: Parameter $k$.

`firing (str)`: Type of spiking function, defaults to `heaviside`, i.e., the Heaviside step function.

`hard_reset (bool)`: Whether to choose hard reset to zero (True) or subtractive reset (False).

### Returns

`o (torch.Tensor)`: Output spikes $O^{l}$ with shape `[T, B, ...]`.

`h (torch.Tensor)`: Final historical potential $H^{l}(T)$ with shape `[B, ...]`.

### Example Usage

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

Multi-step function for the LIM neuron. Unlike the LIF neuron, the LIM neuron outputs the somatic potential at each time step.

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

### Parameters

`x (torch.Tensor)`: Input potential $X^{l}$ with shape `[T, B, ...]`.

`h (torch.Tensor)`: Initial historical potential $H^{l}(0)$ with shape `[B, ...]`.

`u_threshold (torch.Tensor)`: Threshold potential.

`u_rest (torch.Tensor)`: Resting potential.

`tau_m (torch.Tensor)`: Neuron time constant $\tau_{m}$.

`firing (str)`: Type of spiking function, defaults to `heaviside`, i.e., the Heaviside step function.

`hard_reset (bool)`: Whether to choose hard reset to zero (True) or subtractive reset (False).

### Returns

`u (torch.Tensor)`: Output potential $U^{l}$ with shape `[T, B, ...]`.

`h (torch.Tensor)`: Final historical potential $H^{l}(T)$ with shape `[B, ...]`.

### Example Usage

```python
import torch
import matterhorn_pytorch.snn.functional as SF


x = torch.rand(8, 2, 3) * 2
h = torch.rand(2, 3)
y, h = SF.lim_neuron(x, h, torch.tensor(1.0), torch.tensor(0.0), torch.tensor(2.0))
print(y)
print(h)
```

## References

[1] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.

[2] Jiang C, Zhang Y. Klif: An optimized spiking neuron unit for tuning surrogate gradient slope and membrane potential[J]. arXiv preprint arXiv:2302.09238, 2023.