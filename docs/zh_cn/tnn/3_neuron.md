# `matterhorn_pytorch.tnn.response` 与 `matterhorn_pytorch.tnn.neuron`

[回到 `matterhorn_pytorch.tnn`](./README.md)

[English](../../en_us/tnn/3_neuron.md)

[中文](../../zh_cn/tnn/3_neuron.md)

## 模块简介

这两个模块实现 Smith 书第 6.6 节 **Model II 兴奋性神经元**：

1. 各输入线产生分段线性复合响应 $E_i(t, w_i)$；
2. 膜电位为响应之和 $P(t) = \sum_i E_i(t - t_i, w_i)$；
3. 输出脉冲时间 $z = \min\{t \mid P(t) \ge \theta\}$，若无交叉则输出 $\infty$。

训练模式下，`ExcitatoryNeuron` 通过替代梯度（高斯窗）对权重进行反向传播。

## `matterhorn_pytorch.tnn.compound_response`

计算分段线性复合兴奋性响应 $E_i(t, w_i)$（表 6.1、图 6.14）。

```python
compound_response(
    t_rel: torch.Tensor,
    weight: torch.Tensor,
    tr: float,
    tf: float,
    gamma: float
) -> torch.Tensor
```

### 参数

`t_rel (torch.Tensor)`：相对时间（自输入脉冲起算），任意形状。

`weight (torch.Tensor)`：突触权重 $w_i$，可与 `t_rel` 广播。

`tr (float)`：上升时间 $T_r$。

`tf (float)`：下降时间 $T_f$。

`gamma (float)`：缩放因子 $\gamma = T_{max}/W_{max}$。

### 返回值

`response (torch.Tensor)`：响应幅值，与 `t_rel` 同形。

## `matterhorn_pytorch.tnn.membrane_potential`

计算膜电位轨迹：

$$P(t) = \sum_i E_i(t - t_i, w_i)$$

```python
membrane_potential(
    times: torch.Tensor,
    spike_times: torch.Tensor,
    weights: torch.Tensor,
    tr: float,
    tf: float,
    gamma: float,
    connectivity: torch.Tensor = None
) -> torch.Tensor
```

### 参数

`times (torch.Tensor)`：采样时刻，形状 `[T]`。

`spike_times (torch.Tensor)`：输入脉冲时间，形状 `[B, I]`。

`weights (torch.Tensor)`：突触权重，形状 `[B, I]` 或 `[I]`。

`tr, tf, gamma (float)`：响应参数。

`connectivity (torch.Tensor | None)`：连接掩码 `[I]` 或 `[B, I]`，`0` 表示断开。

### 返回值

`potential (torch.Tensor)`：膜电位 $P(t)$，形状 `[B, T]`。

## `matterhorn_pytorch.tnn.ExcitatoryNeuron`

兴奋性神经元（Model II）。输入为各突触首次脉冲时间戳（时间簇），输出为神经元首次发放时间。

```python
ExcitatoryNeuron(
    num_inputs: int,
    w_max: float = 4.0,
    t_max: float = 16.0,
    tr: float = 10.0,
    tf: float = 40.0,
    theta: float = 19.0,
    num_steps: int = 256,
    normalize_input: bool = True,
    connectivity: torch.Tensor = None,
    device: torch.device = None,
    dtype: torch.dtype = None
)
```

### 构造函数参数

`num_inputs (int)`：输入线数 $M$。

`w_max (float)`：最大权重 $W_{max}$。

`t_max (float)`：时间窗 $T_{max}$。

`tr (float)`：响应上升时间 $T_r$。

`tf (float)`：响应下降时间 $T_f$。

`theta (float)`：膜电位阈值 $\theta$。

`num_steps (int)`：膜电位轨迹离散采样点数。

`normalize_input (bool)`：是否对输入簇做局部归一化。

`connectivity (torch.Tensor | None)`：输入连接掩码 `[M]`。

`device (torch.device)`：计算设备。

`dtype (torch.dtype)`：数据类型。

### 可训练参数

`weight (nn.Parameter)`：突触权重，形状 `[M]`。

### 缓冲区

`connectivity (torch.Tensor)`：连接掩码 `[M]`。

### 可重载的方法

#### `forward(self, spike_times: torch.Tensor, global_times: bool = True) -> Tuple[torch.Tensor, torch.Tensor]`

`spike_times (torch.Tensor)`：输入脉冲时间戳 `[B, M]`，无脉冲为 `inf`。

`global_times (bool)`：输入/输出是否为全局时间；`True` 时内部做归一化/反归一化。

返回值：

- `out_time (torch.Tensor)`：输出脉冲时间 `[B]`；
- `t_min (torch.Tensor)`：本次输入簇最早脉冲时间 `[B]`。

### 示例用法

```python
import torch
from matterhorn_pytorch.tnn import ExcitatoryNeuron

neuron = ExcitatoryNeuron(num_inputs=4, theta=5.0, num_steps=128)
neuron.weight.data.fill_(2.0)
neuron.eval()

spikes = torch.tensor([[0.5, 1.0, float("inf"), 2.0]])
out_time, t_min = neuron(spikes)
print(out_time)  # 首次阈值交叉时刻
```

### 监督训练

将神经元置于 `train()` 模式后，可对 `weight` 反向传播：

```python
neuron.train()
out, _ = neuron(spikes)
loss = out.sum()
loss.backward()
print(neuron.weight.grad)
```

也可使用 [`TNNTrainer`](./6_stdp_training.md) 进行端到端优化。
