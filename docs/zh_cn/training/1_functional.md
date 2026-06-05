# `matterhorn_pytorch.training.functional`

[回到 `matterhorn_pytorch.training`](./README.md)

[English](../../en_us/training/1_functional.md)

[中文](../../zh_cn/training/1_functional.md)

## 模块简介

该模块含有 `matterhorn_pytorch.training` 中用于自定义训练的 STDP 相关函数，被 `matterhorn_pytorch.snn.layer` 中的 `STDPLinear` 与 `STDPConv2d` 调用。

## `matterhorn_pytorch.training.functional.stdp_online`

单时间步的在线脉冲时序依赖可塑性（STDP）更新。根据当前输入与输出脉冲更新权重增量及输入/输出迹。

```python
stdp_online(
    delta_weight: torch.Tensor,
    input_trace: torch.Tensor,
    output_trace: torch.Tensor,
    input_spike_train: torch.Tensor,
    output_spike_train: torch.Tensor,
    a_pos: float,
    tau_pos: float,
    a_neg: float,
    tau_neg: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

### 参数

`delta_weight (torch.Tensor)`：权重增量矩阵 $\Delta w$，形状为 `[O, I]`。

`input_trace (torch.Tensor)`：输入迹，形状为 `[B, O, I]`。

`output_trace (torch.Tensor)`：输出迹，形状为 `[B, O, I]`。

`input_spike_train (torch.Tensor)`：当前时间步的输入脉冲，形状为 `[B, I]`。

`output_spike_train (torch.Tensor)`：当前时间步的输出脉冲，形状为 `[B, O]`。

`a_pos (float)`：STDP 参数 $A_{+}$。

`tau_pos (float)`：STDP 参数 $\tau_{+}$。

`a_neg (float)`：STDP 参数 $A_{-}$。

`tau_neg (float)`：STDP 参数 $\tau_{-}$。

### 返回值

`delta_weight (torch.Tensor)`：更新后的权重增量矩阵，形状为 `[O, I]`。

`input_trace (torch.Tensor)`：更新后的输入迹，形状为 `[B, O, I]`。

`output_trace (torch.Tensor)`：更新后的输出迹，形状为 `[B, O, I]`。

## `matterhorn_pytorch.training.functional.stdp_linear`

基于 STDP 的全连接层。执行突触线性变换、胞体动力学与 STDP 权重更新。

```python
stdp_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    soma: Callable,
    hidden: Optional[torch.Tensor] = None,
    input_trace: Optional[torch.Tensor] = None,
    output_trace: Optional[torch.Tensor] = None,
    a_pos: float = 0.015,
    tau_pos: float = 2.0,
    a_neg: float = 0.015,
    tau_neg: float = 2.0,
    training: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

### 参数

`input (torch.Tensor)`：输入脉冲序列，形状为 `[T, B, I]`。

`weight (torch.Tensor)`：权重矩阵，形状为 `[O, I]`。

`soma (Callable)`：胞体模块（如 `snn.IF`），将膜电位映射为脉冲。

`hidden (torch.Tensor | None)`：胞体的历史膜电位。

`input_trace (torch.Tensor | None)`：输入迹。为 `None` 时初始化为零。

`output_trace (torch.Tensor | None)`：输出迹。为 `None` 时初始化为零。

`a_pos`、`tau_pos`、`a_neg`、`tau_neg (float)`：STDP 参数。

`training (bool)`：是否计算 STDP 权重更新。

### 返回值

`output (torch.Tensor)`：输出脉冲序列。

`hidden (torch.Tensor)`：更新后的膜电位。

`input_trace (torch.Tensor)`：更新后的输入迹。

`output_trace (torch.Tensor)`：更新后的输出迹。

## `matterhorn_pytorch.training.functional.stdp_conv2d`

基于 STDP 的二维卷积层。接口与 `stdp_linear` 相同，并额外包含卷积参数。

```python
stdp_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    soma: Callable,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    hidden: Optional[torch.Tensor] = None,
    input_trace: Optional[torch.Tensor] = None,
    output_trace: Optional[torch.Tensor] = None,
    a_pos: float = 0.015,
    tau_pos: float = 2.0,
    a_neg: float = 0.015,
    tau_neg: float = 2.0,
    training: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

### 参数

`input (torch.Tensor)`：输入脉冲序列，形状为 `[T, B, CI, HI, WI]`。

`weight (torch.Tensor)`：卷积核，形状为 `[CO, CI, KH, KW]`。

`soma (Callable)`：胞体模块。

`stride (Tuple[int, int])`：卷积步长。

`padding (Tuple[int, int])`：卷积填充。

`dilation (Tuple[int, int])`：卷积膨胀。

其余参数与 `stdp_linear` 相同。

### 返回值

与 `stdp_linear` 相同。

### 示例用法

```python
import torch
import matterhorn_pytorch.snn as snn
from matterhorn_pytorch.training.functional import stdp_online

delta_weight = torch.zeros(4, 6)
input_trace = torch.zeros(1, 4, 6)
output_trace = torch.zeros(1, 4, 6)
delta_weight, input_trace, output_trace = stdp_online(
    delta_weight,
    input_trace,
    output_trace,
    torch.randint(0, 2, (1, 6)).float(),
    torch.randint(0, 2, (1, 4)).float(),
    0.2, 2.0, 0.2, 2.0
)
print(delta_weight)
```
