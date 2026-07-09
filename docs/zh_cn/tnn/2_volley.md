# `matterhorn_pytorch.tnn.volley`

[回到 `matterhorn_pytorch.tnn`](./README.md)

[English](../../en_us/tnn/2_volley.md)

[中文](../../zh_cn/tnn/2_volley.md)

## 模块简介

该模块实现 TNN 中的**脉冲簇（Volley）**编解码与时空变换工具。在 Smith 的 TNN 框架中，一条输入线用**首次脉冲时间戳** $t_i$ 表示信息；多条输入线组成**时间簇**（time volley）。无脉冲的线记为 $\infty$（代码中用 `float("inf")` 表示）。

参考文献：Smith J E. *Space-Time Computing with Temporal Neural Networks*，第 6.2 节（脉冲编码）与第 7.3 节（OnOff 编码）。

## `matterhorn_pytorch.tnn.EMPTY_SPIKE`

无脉冲哨兵常量，值为 `float("inf")`。

## `matterhorn_pytorch.tnn.is_spike_present`

判断时间戳张量中哪些位置存在有效脉冲。

```python
is_spike_present(t: torch.Tensor) -> torch.Tensor
```

### 参数

`t (torch.Tensor)`：脉冲首次发放时间戳，任意形状。

### 返回值

`mask (torch.Tensor)`：布尔掩码，`True` 表示该线有脉冲。

## `matterhorn_pytorch.tnn.normalize_time_volley`

将时间簇归一化到局部时间参考系（最早脉冲时刻为 0）。对应 Smith 书第 7.4.1 节评估模式中的 **normalize** 步骤。

$$t_i^{norm} = t_i - \min_j\{t_j\}$$

```python
normalize_time_volley(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

### 参数

`t (torch.Tensor)`：时间簇，形状 `[..., M]`，无脉冲处为 `EMPTY_SPIKE`。

### 返回值

`t_norm (torch.Tensor)`：归一化后的时间簇。

`t_min (torch.Tensor)`：各簇最早脉冲的全局时间，形状为 `t` 去掉最后一维后的形状。

### 示例用法

```python
import torch
import matterhorn_pytorch.tnn as TNN

t = torch.tensor([[1.0, 3.0, float("inf")]])
t_norm, t_min = TNN.normalize_time_volley(t)
print(t_norm)   # [[0.0, 2.0, inf]]
print(t_min)    # [1.0]
```

## `matterhorn_pytorch.tnn.denormalize_time_volley`

将局部时间簇还原为全局时间参考系。

$$t_i = t_i^{norm} + t_{min}$$

```python
denormalize_time_volley(t_norm: torch.Tensor, t_min: torch.Tensor) -> torch.Tensor
```

### 参数

`t_norm (torch.Tensor)`：归一化时间簇，形状 `[..., M]`。

`t_min (torch.Tensor)`：最早脉冲全局时间，形状 `[...]` 或 `[..., 1]`。

### 返回值

`t (torch.Tensor)`：全局时间簇。

## `matterhorn_pytorch.tnn.value_to_time_volley`

值簇转时间簇。编码关系为：

$$v_i = \alpha(T_{max} - t_i), \quad \alpha = \frac{V_{max}}{T_{max}}$$

```python
value_to_time_volley(v: torch.Tensor, t_max: float, v_max: float) -> torch.Tensor
```

### 参数

`v (torch.Tensor)`：值簇，形状 `[..., M]`，无值处为 `EMPTY_SPIKE`。

`t_max (float)`：最大时间窗 $T_{max}$。

`v_max (float)`：最大值 $V_{max}$。

### 返回值

`t (torch.Tensor)`：时间簇。

## `matterhorn_pytorch.tnn.time_to_value_volley`

时间簇转值簇，为 `value_to_time_volley` 的逆变换。

```python
time_to_value_volley(t: torch.Tensor, t_max: float, v_max: float) -> torch.Tensor
```

### 参数

`t (torch.Tensor)`：时间簇。

`t_max (float)`：$T_{max}$。

`v_max (float)`：$V_{max}$。

### 返回值

`v (torch.Tensor)`：值簇。

## `matterhorn_pytorch.tnn.merge_time_volleys`

沿最后一维拼接多个时间簇，用于认知柱输入 bundle 的合并。

```python
merge_time_volleys(*volleys: torch.Tensor) -> torch.Tensor
```

### 参数

`*volleys (torch.Tensor)`：若干时间簇，batch 维需一致。

### 返回值

`merged (torch.Tensor)`：合并后的时间簇。

## `matterhorn_pytorch.tnn.onoff_encode`

OnOff 编码：将原始特征向量与其负片拼接，消除簇间偏序关系。对应 Smith 书第 7.3 节：

$$x_{m+k} = I_{max} - x_k$$

```python
onoff_encode(values: torch.Tensor, v_max: float) -> torch.Tensor
```

### 参数

`values (torch.Tensor)`：原始值向量，形状 `[..., M]`，取值 $[0, v_{max}]$。

`v_max (float)`：最大值 $I_{max}/V_{max}$。

### 返回值

`encoded (torch.Tensor)`：OnOff 编码后的向量，形状 `[..., 2M]`。

## `matterhorn_pytorch.tnn.OnOffEncoder`

OnOff 编码模块，将模拟值向量编码为可转为时间簇的 OnOff 值向量。

```python
OnOffEncoder(v_max: float = 1.0)
```

### 构造函数参数

`v_max (float)`：特征最大值 $V_{max}/I_{max}$。

### 可重载的方法

#### `forward(self, x: torch.Tensor) -> torch.Tensor`

`x (torch.Tensor)`：输入特征，形状 `[B, M]`。

返回值：OnOff 编码特征，形状 `[B, 2M]`。

### 示例用法

```python
import torch
from matterhorn_pytorch.tnn import OnOffEncoder, value_to_time_volley

encoder = OnOffEncoder(v_max=4.0)
values = torch.tensor([[1.0, 3.0]])
encoded = encoder(values)
times = value_to_time_volley(encoded, t_max=16.0, v_max=4.0)
```
