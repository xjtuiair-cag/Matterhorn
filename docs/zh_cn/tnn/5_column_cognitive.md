# `matterhorn_pytorch.tnn.column`、`cognitive` 与 `container`

[回到 `matterhorn_pytorch.tnn`](./README.md)

[English](../../en_us/tnn/5_column_cognitive.md)

[中文](../../zh_cn/tnn/5_column_cognitive.md)

## 模块简介

这三个模块构成 TNN 的层级结构：

| 模块 | Smith 书对应 | 作用 |
|------|-------------|------|
| `ExcitatoryColumn` | 兴奋柱 EC（第 7.4 节） | 并行兴奋性神经元阵列 |
| `CognitiveColumn` | 认知柱 CC（第 7.2 节） | FFI → EC → LI 流水线 |
| `Sequential` | 多层 TNN 网络 | 串联多个认知柱或自定义模块 |

认知柱的定义：

$$\text{CC} = \text{FFI} + \text{EC} + \text{LI}$$

处理流程：

```
合并输入 → 归一化 → FFI → EC → LI → 反归一化 → 输出
```

## `matterhorn_pytorch.tnn.build_random_connectivity`

为兴奋柱构建伪随机片内连接（Smith 书第 9.5 节 $50 \times 25 \times 24$ 网络）。

```python
build_random_connectivity(
    num_inputs: int,
    num_neurons: int,
    inputs_per_neuron: int,
    generator: torch.Generator = None
) -> torch.Tensor
```

### 参数

`num_inputs (int)`：输入线数。

`num_neurons (int)`：神经元数。

`inputs_per_neuron (int)`：每个神经元连接的输入线数。

`generator (torch.Generator | None)`：随机数生成器。

### 返回值

`mask (torch.Tensor)`：连接掩码 `[num_neurons, num_inputs]`。

## `matterhorn_pytorch.tnn.ExcitatoryColumn`

兴奋柱：由多个并行兴奋性神经元组成，各神经元拥有独立的片内连接掩码与权重。

```python
ExcitatoryColumn(
    num_inputs: int,
    num_neurons: int,
    inputs_per_neuron: int = None,
    connectivity: torch.Tensor = None,
    w_max: float = 4.0,
    t_max: float = 16.0,
    tr: float = 10.0,
    tf: float = 40.0,
    theta: float = 19.0,
    num_steps: int = 256,
    seed: int = None,
    device: torch.device = None,
    dtype: torch.dtype = None
)
```

### 构造函数参数

`num_inputs (int)`：输入线数。

`num_neurons (int)`：柱内兴奋性神经元数。

`inputs_per_neuron (int | None)`：每神经元连接数，默认 `num_inputs // 2`。

`connectivity (torch.Tensor | None)`：自定义连接掩码 `[N, I]`。

`w_max, t_max, tr, tf, theta, num_steps`：兴奋性神经元参数，详见 [`ExcitatoryNeuron`](./3_neuron.md)。

`seed (int | None)`：随机连接种子。

`device, dtype`：张量工厂参数。

### 子模块

`neurons (nn.ModuleList)`：柱内 `ExcitatoryNeuron` 列表。

### 可重载的方法

#### `forward(self, spike_times: torch.Tensor, global_times: bool = True) -> Tuple[torch.Tensor, torch.Tensor]`

`spike_times (torch.Tensor)`：输入时间簇 `[B, I]`。

`global_times (bool)`：输入/输出是否为全局时间参考系。

返回值：

- `out_volley (torch.Tensor)`：输出时间簇 `[B, N]`；
- `t_min (torch.Tensor)`：输入簇最早脉冲时间 `[B]`。

## `matterhorn_pytorch.tnn.CognitiveColumn`

认知柱：由 FFI、EC、LI 组成的基本 TNN 计算单元。

```python
CognitiveColumn(
    num_inputs: int,
    num_neurons: int,
    inputs_per_neuron: int = None,
    connectivity: torch.Tensor = None,
    t_f: float = 1.3,
    k_f: int = 10,
    t_l: float = 5.0,
    k_l: int = 3,
    w_max: float = 4.0,
    t_max: float = 16.0,
    tr: float = 10.0,
    tf: float = 40.0,
    theta: float = 19.0,
    num_steps: int = 256,
    seed: int = None,
    device: torch.device = None,
    dtype: torch.dtype = None
)
```

### 构造函数参数

`num_inputs (int)`：输入线数（合并后 bundle 宽度）。

`num_neurons (int)`：兴奋柱神经元数。

`inputs_per_neuron, connectivity, seed`：兴奋柱连接参数。

`t_f, k_f`：FFI 参数，详见 [`inhibition`](./4_inhibition.md)。

`t_l, k_l`：LI 参数。

其余参数：兴奋性神经元参数。

### 子模块

`ffi (FeedforwardInhibition)`：前馈抑制柱。

`ec (ExcitatoryColumn)`：兴奋柱。

`li (LateralInhibition)`：侧向抑制柱。

### 可重载的方法

#### `forward(self, spike_times=None, merged_input=None) -> Tuple[torch.Tensor, dict]`

`spike_times (Iterable[torch.Tensor] | None)`：待合并的多个输入时间簇。

`merged_input (torch.Tensor | None)`：已合并的输入时间簇 `[B, I]`。

二者至少提供一个。

返回值：

- `output (torch.Tensor)`：输出时间簇 `[B, N]`；
- `states (dict)`：中间状态，含 `merged_input`、`ffi_out`、`ec_out`、`li_out`、`output`。

### 示例用法

```python
import torch
from matterhorn_pytorch.tnn import CognitiveColumn, Sequential

cc = CognitiveColumn(num_inputs=4, num_neurons=2, theta=5.0, seed=0)
for n in cc.ec.neurons:
    n.weight.data.fill_(2.0)
cc.eval()

merged = torch.tensor([[1.0, 1.2, 1.5, float("inf")]])
out, states = cc(merged_input=merged)
print(out.shape)  # [1, 2]

# 多层串联
net = Sequential(cc, cc)
y = net(merged)
```

## `matterhorn_pytorch.tnn.Sequential`

TNN 序列容器，支持认知柱及任意 `nn.Module` 串联。

```python
Sequential(*args: nn.Module, return_states: bool = False)
```

### 构造函数参数

`*args (nn.Module)`：按顺序排列的模块。

`return_states (bool)`：是否返回各层中间状态。

### 可重载的方法

#### `forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor | Tuple[torch.Tensor, list]`

`x (torch.Tensor)`：输入时间簇 `[B, M]`。

对 `CognitiveColumn` 子模块，自动以 `merged_input=x` 调用。

返回值：最终输出时间簇；若 `return_states=True`，则额外返回各层状态列表。
