# `matterhorn_pytorch.tnn.stdp` 与 `matterhorn_pytorch.tnn.training`

[回到 `matterhorn_pytorch.tnn`](./README.md)

[English](../../en_us/tnn/6_stdp_training.md)

[中文](../../zh_cn/tnn/6_stdp_training.md)

## 模块简介

这两个模块提供 TNN 的**训练**能力，对应 Smith 书中的两类学习机制：

1. **STDP**（第 6.4.3 节）：基于输入/输出脉冲时间差的局部无监督权重更新；
2. **Model II 均值法**（第 6.6.2 节）：$w_i = (T_{max} - \bar{t}_i) / \gamma$；
3. **监督学习**：以输出脉冲时间为目标，通过替代梯度反向传播优化权重。

## `matterhorn_pytorch.tnn.stdp_gate`

分段线性 STDP 门控函数 $G(\Delta t)$，其中 $\Delta t = t_{in} - t_{out}$：

- 输入先于输出（$\Delta t < 0$）→ 增强（$+1$）；
- 输入后于输出（$\Delta t > 0$）→ 减弱（$-1$）。

```python
stdp_gate(
    delta_t: torch.Tensor,
    t_max: float,
    slope: float = 0.1
) -> torch.Tensor
```

### 参数

`delta_t (torch.Tensor)`：时间差 $\Delta t$。

`t_max (float)`：编码窗 $T_{max}$，用于远端饱和。

`slope (float)`：$\Delta t = 0$ 附近的线性斜率。

### 返回值

`gate (torch.Tensor)`：更新方向与幅度。

## `matterhorn_pytorch.tnn.volley_stdp_update`

基于脉冲簇的局部 STDP 权重更新（Smith 书第 6.4.3 节）。

```python
volley_stdp_update(
    weights: torch.Tensor,
    input_times: torch.Tensor,
    output_time: torch.Tensor,
    learning_rate: float = 0.05,
    t_max: float = 16.0,
    w_max: float = 4.0,
    connectivity: torch.Tensor = None
) -> torch.Tensor
```

### 参数

`weights (torch.Tensor)`：突触权重 `[I]` 或 `[O, I]`。

`input_times (torch.Tensor)`：输入脉冲时间 `[B, I]` 或 `[I]`（归一化参考系）。

`output_time (torch.Tensor)`：输出脉冲时间标量或 `[B]`。

`learning_rate (float)`：学习率 $\lambda$。

`t_max (float)`：$T_{max}$。

`w_max (float)`：$W_{max}$。

`connectivity (torch.Tensor | None)`：连接掩码。

### 返回值

`new_weights (torch.Tensor)`：更新后的权重，裁剪至 $[0, W_{max}]$。

## `matterhorn_pytorch.tnn.mean_time_to_weight`

Model II 非 STDP 训练：

$$w_i = \frac{T_{max} - \bar{t}_i}{\gamma}$$

```python
mean_time_to_weight(
    mean_times: torch.Tensor,
    t_max: float,
    gamma: float
) -> torch.Tensor
```

### 参数

`mean_times (torch.Tensor)`：各输入线平均脉冲时间。

`t_max (float)`：$T_{max}$。

`gamma (float)`：$\gamma = T_{max}/W_{max}$。

### 返回值

`weights (torch.Tensor)`：训练后权重。

## `matterhorn_pytorch.tnn.spike_time_mse_loss`

输出脉冲时间的均方误差损失。无脉冲位置用大罚值替代。

```python
spike_time_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fill: float = 1e6
) -> torch.Tensor
```

### 参数

`pred (torch.Tensor)`：预测时间 `[B]` 或 `[B, N]`。

`target (torch.Tensor)`：目标时间。

`fill (float)`：无脉冲时的替代值。

### 返回值

`loss (torch.Tensor)`：标量损失。

## `matterhorn_pytorch.tnn.train_excitatory_neuron_stdp`

使用 volley STDP 无监督训练单个兴奋性神经元（原地修改权重）。

```python
train_excitatory_neuron_stdp(
    neuron: ExcitatoryNeuron,
    training_volleys: Iterable[torch.Tensor],
    learning_rate: float = 0.05,
    epochs: int = 1
) -> ExcitatoryNeuron
```

### 参数

`neuron (ExcitatoryNeuron)`：待训练神经元。

`training_volleys (Iterable[torch.Tensor])`：训练时间簇序列 `[B, I]`。

`learning_rate (float)`：学习率。

`epochs (int)`：遍历训练集次数。

### 返回值

训练后的神经元（原地修改）。

## `matterhorn_pytorch.tnn.train_column_mean_weights`

使用 Model II 均值法为兴奋柱中每个神经元设置权重。

```python
train_column_mean_weights(
    column: ExcitatoryColumn,
    training_volleys: torch.Tensor
) -> ExcitatoryColumn
```

### 参数

`column (ExcitatoryColumn)`：兴奋柱。

`training_volleys (torch.Tensor)`：训练时间簇 `[S, I]`。

### 返回值

训练后的兴奋柱（原地修改）。

## `matterhorn_pytorch.tnn.TNNTrainer`

TNN 监督训练器，对含 `ExcitatoryNeuron` / `ExcitatoryColumn` / `CognitiveColumn` 的模型做 Adam 优化。

```python
TNNTrainer(model: nn.Module, learning_rate: float = 1e-2)
```

### 构造函数参数

`model (nn.Module)`：待训练 TNN 模型。

`learning_rate (float)`：Adam 学习率。

### 方法

#### `supervised_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float`

`inputs (torch.Tensor)`：输入时间簇 `[B, I]`。

`targets (torch.Tensor)`：目标输出时间 `[B]` 或 `[B, N]`。

返回值：本步损失值（`float`）。

### 示例用法

```python
import torch
from matterhorn_pytorch.tnn import ExcitatoryNeuron, TNNTrainer

neuron = ExcitatoryNeuron(num_inputs=3, theta=3.0, num_steps=64)
trainer = TNNTrainer(neuron, learning_rate=1e-2)

x = torch.tensor([[0.1, 0.5, 1.0]])
neuron.eval()
with torch.no_grad():
    target, _ = neuron(x)

for _ in range(10):
    loss = trainer.supervised_step(x, target)
    print(loss)
```
