# `matterhorn_pytorch.snn.encoder`

[回到 `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/8_encoder.md)

[中文](../../zh_cn/snn/8_encoder.md)

## 模块简介

SNNs 的编码器，将模拟值编码为脉冲序列。

## `matterhorn_pytorch.snn.DirectEncoder` / `matterhorn_pytorch.snn.encoder.Direct`

直接编码的编码器。将数据从 `DataLoader` 中提取出来后，数据的第一位是批，即数据的形状为 `[B, T, ...]` ，此时需要通过该编码器转置前两个维度，即将形状由 `[B, T, ...]` 改变为 `[T, B, ...]` ，以使 SNN 模型正常工作。

```python
DirectEncoder()
```

## 示例用法

```python
import torch
import matterhorn_pytorch as mth


data = torch.rand(4, 16, 1, 28, 28) # [B = 4, T = 16, C, H, W]
print(data.shape)
encoder = mth.snn.DirectEncoder()
res = encoder(data)
print(res.shape)
```

## `matterhorn_pytorch.snn.PoissionEncoder` / `matterhorn_pytorch.snn.encoder.Poission`

泊松编码（速率编码）的编码器。将值转换为输入层每个神经元发放脉冲的概率。可以用如下公式描述：

$$O_{i}^{0}(t) = rand(t) \le X_{i} ? 1 : 0$$

其中 $rand(\cdot)$ 为随机数函数，生成 $[0,1)$ 区间内的随机数。

```python
PoissonEncoder(
    time_steps: int = 1
)
```

## 构造函数参数

`time_steps (int)` ：经过泊松编码后，生成的张量时间步长 `T` 。

## 示例用法

```python
import torch
import matterhorn_pytorch as mth


data = torch.rand(4, 1, 28, 28) # [B = 4, C, H, W]
print(data.shape)
encoder = mth.snn.PoissionEncoder(32)
res = encoder(data)
print(res.shape)
```

原图像：

![编码前的图像](../../../assets/docs/snn/encoder_1.png)

编码后经过平均脉冲发射率解码后的图像：

![编码后的图像](../../../assets/docs/snn/encoder_2.png)

## `matterhorn_pytorch.snn.TemporalEncoder` / `matterhorn_pytorch.snn.encoder.Temporal`

时间编码的编码器。在时间步超过给定值后，以一定概率 $p$ 产生脉冲。可以用如下公式描述：

$$O_{i}^{0}(t) = (t \ge X_{i}) \times (rand(t) \le p) ? 1 : 0$$

其中 $rand(\cdot)$ 为随机数函数，生成 $[0,1)$ 区间内的随机数。

```python
Temporal(
    time_steps: int = 1,
    prob: float = 1.0,
    transform: Callable = lambda x: x,
    reset_after_process: bool = True
)
```

## 构造函数参数

`time_steps (int)` ：经过时间编码后，生成的张量时间步长 `T` 。

`prob (float)` ：时间步超过给定值时，生成脉冲的概率 $p$ 。

`transform (Callable)` ：将数据 $X$ 如何变换。

`reset_after_process (bool)` ：是否在执行完后自动重置，若为 `False` 则需要手动重置。

## 示例用法

```python
import torch
import matterhorn_pytorch as mth


data = torch.rand(4, 1, 28, 28) # [B = 4, C, H, W]
print(data.shape)
encoder = mth.snn.TemporalEncoder(32, transform = lambda x: 32 * x)
res = encoder(data)
print(res.shape)
```

原图像：

![编码前的图像](../../../assets/docs/snn/encoder_1.png)

编码后经过平均脉冲发射率解码后的图像：

![编码后的图像](../../../assets/docs/snn/encoder_3.png)

编码后经过最早到达脉冲时间步解码后的图像：

![编码后的图像](../../../assets/docs/snn/encoder_4.png)