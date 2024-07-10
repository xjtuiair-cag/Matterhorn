# `matterhorn_pytorch.snn.decoder`

[回到 `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/9_decoder.md)

[中文](../../zh_cn/snn/9_decoder.md)

## 模块简介

SNNs 的解码器，将脉冲序列解码为模拟值。

## `matterhorn_pytorch.snn.SumSpikeDecoder` / `matterhorn_pytorch.snn.decoder.SumSpike`

总脉冲数解码器。遵循以下公式：

$$Y_{i}=\sum_{t=1}^{T}{O_{i}^{L}(t)}$$

其中 $T$ 为时间步的大小（脉冲序列的第一个维度）。

```python
SumSpike()
```

### 示例用法

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

平均脉冲发射率解码器。遵循以下公式：

$$Y_{i}=\frac{1}{T}\sum_{t=1}^{T}{O_{i}^{L}(t)}$$

其中 $T$ 为时间步的大小（脉冲序列的第一个维度）。

```python
AverageSpike()
```

### 示例用法

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

最早到达时间解码器。遵循以下公式：

$$Y_{i}=min(\{t_{i}|O_{i}^{L}(t_{i})=1\})$$

```python
MinTime(
    empty_fill: float = -1,
    transform: Callable = lambda x: x
)
```

### 构造函数参数

`empty_fill (float)` ：当某个空间位置的脉冲序列无脉冲时，其时间应当用何值表示。一般为 `-1` 。在 TNN 模型中，您可能需要将其设置为 `torch.inf` 。

`transform (Callable)` ：将结果 $y$ 如何变换。

### 示例用法

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

最早到达时间解码器。遵循以下公式：

$$Y_{i}=\frac{1}{S}\sum_{S}\{t_{i}|O_{i}^{L}(t_{i})=1\}$$

其中 $S=\sum_{t=1}^{T}{O_{i}^{L}(t)}$ 为神经元 $i$ 的脉冲总数。

```python
AverageTime(
    empty_fill: float = -1,
    transform: Callable = lambda x: x
)
```

### 构造函数参数

`empty_fill (float)` ：当某个空间位置的脉冲序列无脉冲时，其时间应当用何值表示。一般为 `-1` 。在 TNN 模型中，您可能需要将其设置为 `torch.inf` 。

`transform (Callable)` ：将结果 $y$ 如何变换。

### 示例用法

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