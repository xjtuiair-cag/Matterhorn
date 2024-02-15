# `matterhorn_pytorch.snn.decoder`

[Back to `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/9_decoder.md)

[中文](../../zh_cn/snn/9_decoder.md)

## Module Introduction

Decoder for SNNs, decoding spike sequences into analog values.

## `matterhorn_pytorch.snn.SumSpikeDecoder` / `matterhorn_pytorch.snn.decoder.SumSpike`

Decoder for spike counts. Follows the formula:

$$Y_{i}=\sum_{t=1}^{T}{O_{i}^{L}(t)}$$

where $T$ is the size of the time step (the first dimension of the spike sequence).

```python
SumSpike()
```

### Example Usage

```python
import torch
import matterhorn_pytorch as mth
import matterhorn_pytorch.snn.functional as SF


data = SF.val_to_spike(torch.rand(8, 1, 6)) # [T, B, ...]
print(data)
decoder = mth.snn.SumSpikeDecoder()
res = decoder(data)
print(res)
```

## `matterhorn_pytorch.snn.AvgSpikeDecoder` / `matterhorn_pytorch.snn.decoder.AverageSpike`

Decoder for average spike rate. Follows the formula:

$$Y_{i}=\frac{1}{T}\sum_{t=1}^{T}{O_{i}^{L}(t)}$$

where $T$ is the size of the time step (the first dimension of the spike sequence).

```python
AverageSpike()
```

### Example Usage

```python
import torch
import matterhorn_pytorch as mth
import matterhorn_pytorch.snn.functional as SF


data = SF.val_to_spike(torch.rand(8, 1, 6)) # [T, B, ...]
print(data)
decoder = mth.snn.AvgSpikeDecoder()
res = decoder(data)
print(res)
```

## `matterhorn_pytorch.snn.MinTimeDecoder` / `matterhorn_pytorch.snn.decoder.MinTime`

Decoder for the earliest arrival time. Follows the formula:

$$Y_{i}=min(\{t_{i}|O_{i}^{L}(t_{i})=1\})$$

```python
MinTime(
    empty_fill: float = -1,
    transform: Callable = lambda x: x,
    reset_after_process: bool = True
)
```

### Constructor Arguments

`empty_fill (float)`: Value to represent when there is no spike at a particular spatial position in the spike sequence. Typically `-1`. In TNN models, you might want to set this to `torch.inf`.

`transform (Callable)`: How to transform the result $y$.

`reset_after_process (bool)`: Whether to reset automatically after execution. If `False`, manual reset is required.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth
import matterhorn_pytorch.snn.functional as SF


data = SF.val_to_spike(torch.rand(8, 1, 6)) # [T, B, ...]
print(data)
decoder = mth.snn.MinTimeDecoder()
res = decoder(data)
print(res)
```

## `matterhorn_pytorch.snn.AvgTimeDecoder` / `matterhorn_pytorch.snn.decoder.AverageTime`

Decoder for the average arrival time. Follows the formula:

$$Y_{i}=\frac{1}{S}\sum_{S}\{t_{i}|O_{i}^{L}(t_{i})=1\}$$

Where $S=\sum_{t=1}^{T}{O_{i}^{L}(t)}$ is count of spikes in neuron $i$.

```python
AverageTime(
    empty_fill: float = -1,
    transform: Callable = lambda x: x,
    reset_after_process: bool = True
)
```

### Constructor Arguments

`empty_fill (float)`: Value to represent when there is no spike at a particular spatial position in the spike sequence. Typically `-1`. In TNN models, you might want to set this to `torch.inf`.

`transform (Callable)`: How to transform the result $y$.

`reset_after_process (bool)`: Whether to reset automatically after execution. If `False`, manual reset is required.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth
import matterhorn_pytorch.snn.functional as SF


data = SF.val_to_spike(torch.rand(8, 1, 6)) # [T, B, ...]
print(data)
decoder = mth.snn.AvgTimeDecoder()
res = decoder(data)
print(res)
```