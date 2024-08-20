# `matterhorn_pytorch.training.functional`

[回到 `matterhorn_pytorch.training`](./README.md)

[English](../../en_us/training/1_functional.md)

[中文](../../zh_cn/training/1_functional.md)

## 模块简介

该模块为 `matterhorn_pytorch.training` 模块的函数库，存储自定义训练的函数。

## `matterhorn_pytorch.training.functional.stdp`

脉冲时序依赖可塑性（STDP）函数。由输入脉冲序列和输出脉冲序列得到权重的更新量。

```python
stdp(
    delta_weight: torch.Tensor,
    input_spike_train: torch.Tensor,
    output_spike_train: torch.Tensor,
    a_pos: float,
    tau_pos: float,
    a_neg: float,
    tau_neg: float,
    precision: float = 1e-6
) -> torch.Tensor
```

### 参数

`delta_weight (torch.Tensor)` ：权重增量矩阵 $\Delta w$ ，大小为 `[O, I]` 。

`input_spike_train (torch.Tensor)` ：该层的输入脉冲序列，大小为 `[T, B, I]` 。

`output_spike_train (torch.Tensor)` ：该层的输出脉冲序列，大小为 `[T, B, O]` 。

`a_pos (float)` ： STDP 参数 $A_{+}$ 。

`tau_pos (float)` ： STDP 参数 $\tau_{+}$ 。

`a_neg (float)` ： STDP 参数 $A_{-}$ 。

`tau_neg (float)` ： STDP 参数 $\tau_{-}$ 。

`precision (float)` ：精度。 STDP 的算法在不同的平台上计算时，会由于精度的不同产生误差。该参数规定了 $\Delta w$ 的最小值，小于这个值的数字会被当做 $0$ 处理。

### 返回值

`delta_weight (torch.Tensor)` ：计算后的权重增量矩阵 $\Delta w$ ，大小为 `[O, I]` 。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth
import matterhorn_pytorch.snn.functional as SF
from matterhorn_pytorch.training.functional import stdp


delta_weight = torch.zeros(4, 6, dtype = torch.float) # [O = 4, I = 6]
input_spike_train = SF.to_spike_train(torch.rand(8, 1, 6)) # [T = 8, B = 1, I = 6]
output_spike_train = SF.to_spike_train(torch.rand(8, 1, 4)) # [T = 8, B = 1, O = 4]
delta_weight = stdp(delta_weight, input_spike_train, output_spike_train, 0.2, 2, 0.2, 2)
print(delta_weight)
```