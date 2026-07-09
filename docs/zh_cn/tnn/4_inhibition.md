# `matterhorn_pytorch.tnn.inhibition`

[回到 `matterhorn_pytorch.tnn`](./README.md)

[English](../../en_us/tnn/4_inhibition.md)

[中文](../../zh_cn/tnn/4_inhibition.md)

## 模块简介

该模块实现 Smith 书第 7.5 节的**抑制柱（Inhibitory Column）**模型。抑制柱在更高抽象层操作，不建模单个抑制神经元，而是对脉冲簇进行过滤。

TNN 认知柱包含两种抑制机制：

- **前馈抑制（FFI, Feedforward Inhibition）**：抑制输入簇中过于密集的脉冲；
- **侧向抑制（LI, Lateral Inhibition）**：tk-WTA，保留时间窗内最早的 $k_L$ 个脉冲。

两种抑制均在**归一化时间参考系**下工作。

## `matterhorn_pytorch.tnn.feedforward_inhibit`

前馈抑制 FFI（Smith 书第 7.5.3 节）。若在首个脉冲后 $t_F$ 时间窗内超过 $k_F$ 个脉冲，则整簇清零（nullify）。

```python
feedforward_inhibit(
    t_volley: torch.Tensor,
    t_f: float,
    k_f: int
) -> torch.Tensor
```

### 参数

`t_volley (torch.Tensor)`：归一化时间簇 `[..., M]`。

`t_f (float)`：时间窗参数 $t_F$。

`k_f (int)`：脉冲数阈值 $k_F$。

### 返回值

`out (torch.Tensor)`：抑制后的时间簇。

### 示例用法

```python
import torch
from matterhorn_pytorch.tnn import feedforward_inhibit

# 窗内脉冲过多 → 整簇清零
t = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
out = feedforward_inhibit(t, t_f=1.3, k_f=10)
```

## `matterhorn_pytorch.tnn.lateral_inhibit`

侧向抑制 LI / tk-WTA（Smith 书第 7.5.2 节）。保留时间 $\le t_L$ 的最早 $k_L$ 个脉冲，其余置空。

```python
lateral_inhibit(
    t_volley: torch.Tensor,
    t_l: float,
    k_l: int
) -> torch.Tensor
```

### 参数

`t_volley (torch.Tensor)`：归一化时间簇 `[..., M]`。

`t_l (float)`：时间窗 $t_L$。

`k_l (int)`：保留脉冲数 $k_L$。

### 返回值

`out (torch.Tensor)`：抑制后的时间簇。

## `matterhorn_pytorch.tnn.FeedforwardInhibition`

前馈抑制柱（FFI）模块。

```python
FeedforwardInhibition(t_f: float = 1.3, k_f: int = 10)
```

### 构造函数参数

`t_f (float)`：时间窗 $t_F$。

`k_f (int)`：脉冲数阈值 $k_F$。

### 可重载的方法

#### `forward(self, t_volley: torch.Tensor, normalized: bool = True) -> torch.Tensor`

`t_volley (torch.Tensor)`：时间簇 `[B, M]`。

`normalized (bool)`：是否为归一化时间簇；`False` 时内部自动归一化并在输出时反归一化。

返回值：抑制后的时间簇。

## `matterhorn_pytorch.tnn.LateralInhibition`

侧向抑制柱（LI）模块。

```python
LateralInhibition(t_l: float = 5.0, k_l: int = 3)
```

### 构造函数参数

`t_l (float)`：时间窗 $t_L$。

`k_l (int)`：保留脉冲数 $k_L$。

### 可重载的方法

#### `forward(self, t_volley: torch.Tensor, normalized: bool = True) -> torch.Tensor`

参数与返回值同 `FeedforwardInhibition.forward`。
