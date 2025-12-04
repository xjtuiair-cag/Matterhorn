# `matterhorn_pytorch.snn.encoder`

[Back to `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/8_encoder.md)

[中文](../../zh_cn/snn/8_encoder.md)

## Module Introduction

Encoder for SNNs, encoding analog values into spike sequences.

## `matterhorn_pytorch.snn.DirectEncoder` / `matterhorn_pytorch.snn.encoder.Direct`

Encoder for direct encoding. After extracting data from the `DataLoader`, where the first dimension represents batches, the shape of the data is `[B, T, ...]`. In this case, this encoder is needed to transpose the first two dimensions, changing the shape from `[B, T, ...]` to `[T, B, ...]`, to make the SNN model work properly.

```python
Direct()
```

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


data = torch.rand(4, 16, 1, 28, 28) # [B = 4, T = 16, C, H, W]
print(data.shape)
encoder = mth.snn.DirectEncoder()
res = encoder(data)
print(res.shape)
```

## `matterhorn_pytorch.snn.AnalogEncoder` / `matterhorn_pytorch.snn.encoder.Analog`

An encoder for analog value encoding. After extracting data from the `DataLoader`, it duplicates the data unchanged to each time step, then outputs analog values with a shape of `[T, B, ...]`.

```python
Analog(
    time_steps: int = 1,
)
```

### Constructor Parameters

`time_steps (int)`: The time step length `T` of the generated tensor after analog encoding.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


data = torch.rand(4, 1, 28, 28) # [B, C, H, W]
print(data.shape)
encoder = mth.snn.AnalogEncoder(16)
res = encoder(data)
print(res.shape)
```

## `matterhorn_pytorch.snn.PoissionEncoder` / `matterhorn_pytorch.snn.encoder.Poission`

Encoder for Poisson encoding (rate encoding). Converts values to the probability of each neuron in the input layer emitting spikes. It can be described by the following formula:

$$O_{i}^{0}(t) = rand(t) \le X_{i} ? 1 : 0$$

where $rand(\cdot)$ is a random number function that generates random numbers in the interval $[0,1)$.

```python
Poisson(
    time_steps: int = 1,
    input_range: _Tuple[float, float] | _Tuple[int, int] | float | int | None = None,
    precision: float = 1e-5,
    count: bool = False
)
```

### Constructor Arguments

`time_steps (int)`: Length of the tensor after Poisson encoding, denoted as `T`.

`input_range (float* | int*)`: The lower and upper bounds of the input, used for normalizing the input to suit Poisson encoding. Defaults to `(0.0, 1.0)`.

`precision (float)`: Since $\lambda > 0$ in a Poisson distribution, this parameter specifies the lower bound for $\lambda`.

`count (bool)`: Whether to send the spike count.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


data = torch.rand(4, 1, 28, 28) # [B, C, H, W]
print(data.shape)
encoder = mth.snn.PoissionEncoder(32)
res = encoder(data)
print(res.shape)
```

Original Image:

![Original Image Before Encoding](../../../assets/docs/snn/encoder_1.png)

Image after encoding and decoding with average spike rate:

![Image After Encoding](../../../assets/docs/snn/encoder_2.png)

## `matterhorn_pytorch.snn.TemporalEncoder` / `matterhorn_pytorch.snn.encoder.Temporal`

Encoder for temporal encoding. After a given time step, spikes are generated with a certain probability $p$. It can be described by the following formula:

$$O_{i}^{0}(t) = (t \ge X_{i}) \times (rand(t) \le p) ? 1 : 0$$

where $rand(\cdot)$ is a random number function that generates random numbers in the interval $[0,1)$.

```python
Temporal(
    time_steps: int = 1,
    prob: float = 1.0,
    transform: Callable = lambda x: x
)
```

### Constructor Arguments

`time_steps (int)`: Length of the tensor after temporal encoding, denoted as `T`.

`prob (float)`: Probability of spike generation $p$ after a given time step.

`transform (Callable)`: How data $X$ should be transformed.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


data = torch.rand(4, 1, 28, 28) # [B = 4, C, H, W]
print(data.shape)
encoder = mth.snn.TemporalEncoder(32, transform = lambda x: 32 * x)
res = encoder(data)
print(res.shape)
```

Original Image:

![Original Image Before Encoding](../../../assets/docs/snn/encoder_1.png)

Image after encoding and decoding with average spike rate:

![Image After Encoding](../../../assets/docs/snn/encoder_3.png)

Image after encoding and decoding with the earliest arrival spike time step:

![Image After Encoding](../../../assets/docs/snn/encoder_4.png)