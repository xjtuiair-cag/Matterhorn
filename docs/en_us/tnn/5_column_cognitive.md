# `matterhorn_pytorch.tnn.column`, `cognitive`, and `container`

[Back to `matterhorn_pytorch.tnn`](./README.md)

[English](../../en_us/tnn/5_column_cognitive.md)

[中文](../../zh_cn/tnn/5_column_cognitive.md)

## Module Introduction

These modules form the hierarchical structure of TNNs:

| Module | Smith Reference | Role |
|--------|-----------------|------|
| `ExcitatoryColumn` | EC (Section 7.4) | Parallel excitatory neuron array |
| `CognitiveColumn` | CC (Section 7.2) | FFI → EC → LI pipeline |
| `Sequential` | Multi-layer TNN | Chains cognitive columns or custom modules |

Cognitive column definition:

$$\text{CC} = \text{FFI} + \text{EC} + \text{LI}$$

Processing flow:

```
merge inputs → normalize → FFI → EC → LI → denormalize → output
```

## `matterhorn_pytorch.tnn.build_random_connectivity`

Builds pseudo-random within-column connectivity (Smith Section 9.5, $50 \times 25 \times 24$ network).

```python
build_random_connectivity(
    num_inputs: int,
    num_neurons: int,
    inputs_per_neuron: int,
    generator: torch.Generator = None
) -> torch.Tensor
```

### Arguments

`num_inputs (int)`: Number of input lines.

`num_neurons (int)`: Number of neurons.

`inputs_per_neuron (int)`: Inputs connected per neuron.

`generator (torch.Generator | None)`: Random generator.

### Returns

`mask (torch.Tensor)`: Connection mask `[num_neurons, num_inputs]`.

## `matterhorn_pytorch.tnn.ExcitatoryColumn`

Excitatory column: multiple parallel excitatory neurons, each with its own connectivity mask and weights.

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

### Constructor Arguments

`num_inputs (int)`: Input line count.

`num_neurons (int)`: Number of excitatory neurons in the column.

`inputs_per_neuron (int | None)`: Connections per neuron; default `num_inputs // 2`.

`connectivity (torch.Tensor | None)`: Custom mask `[N, I]`.

`w_max, t_max, tr, tf, theta, num_steps`: Neuron parameters; see [`ExcitatoryNeuron`](./3_neuron.md).

`seed (int | None)`: Random connectivity seed.

`device, dtype`: Tensor factory arguments.

### Submodules

`neurons (nn.ModuleList)`: List of `ExcitatoryNeuron` instances.

### Overridable Methods

#### `forward(self, spike_times: torch.Tensor, global_times: bool = True) -> Tuple[torch.Tensor, torch.Tensor]`

`spike_times (torch.Tensor)`: Input volley `[B, I]`.

`global_times (bool)`: Whether input/output use global time.

Returns:

- `out_volley (torch.Tensor)`: Output volley `[B, N]`;
- `t_min (torch.Tensor)`: Earliest input spike time `[B]`.

## `matterhorn_pytorch.tnn.CognitiveColumn`

Cognitive column: basic TNN compute unit composed of FFI, EC, and LI.

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

### Constructor Arguments

`num_inputs (int)`: Input width after bundle merge.

`num_neurons (int)`: Excitatory column size.

`inputs_per_neuron, connectivity, seed`: Excitatory column connectivity.

`t_f, k_f`: FFI parameters; see [`inhibition`](./4_inhibition.md).

`t_l, k_l`: LI parameters.

Remaining arguments: excitatory neuron parameters.

### Submodules

`ffi (FeedforwardInhibition)`: Feedforward inhibitory column.

`ec (ExcitatoryColumn)`: Excitatory column.

`li (LateralInhibition)`: Lateral inhibitory column.

### Overridable Methods

#### `forward(self, spike_times=None, merged_input=None) -> Tuple[torch.Tensor, dict]`

`spike_times (Iterable[torch.Tensor] | None)`: Multiple input volleys to merge.

`merged_input (torch.Tensor | None)`: Pre-merged input volley `[B, I]`.

At least one must be provided.

Returns:

- `output (torch.Tensor)`: Output volley `[B, N]`;
- `states (dict)`: Intermediate states: `merged_input`, `ffi_out`, `ec_out`, `li_out`, `output`.

### Example Usage

```python
import torch
from matterhorn_pytorch.tnn import CognitiveColumn, Sequential

cc = CognitiveColumn(num_inputs=4, num_neurons=2, theta=5.0, seed=0)
for n in cc.ec.neurons:
    n.weight.data.fill_(2.0)
cc.eval()

merged = torch.tensor([[1.0, 1.2, 1.5, float("inf")]])
out, states = cc(merged_input=merged)

net = Sequential(cc, cc)
y = net(merged)
```

## `matterhorn_pytorch.tnn.Sequential`

TNN sequence container chaining cognitive columns or any `nn.Module`.

```python
Sequential(*args: nn.Module, return_states: bool = False)
```

### Constructor Arguments

`*args (nn.Module)`: Modules in order.

`return_states (bool)`: Whether to return per-layer states.

### Overridable Methods

#### `forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor | Tuple[torch.Tensor, list]`

`x (torch.Tensor)`: Input volley `[B, M]`.

`CognitiveColumn` submodules are called with `merged_input=x`.

Returns: Final output volley; if `return_states=True`, also returns a list of layer states.
