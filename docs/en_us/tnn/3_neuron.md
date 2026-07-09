# `matterhorn_pytorch.tnn.response` and `matterhorn_pytorch.tnn.neuron`

[Back to `matterhorn_pytorch.tnn`](./README.md)

[English](../../en_us/tnn/3_neuron.md)

[中文](../../zh_cn/tnn/3_neuron.md)

## Module Introduction

These modules implement Smith Section 6.6 **Model II excitatory neurons**:

1. Each input line produces a piecewise-linear compound response $E_i(t, w_i)$;
2. Membrane potential is the sum $P(t) = \sum_i E_i(t - t_i, w_i)$;
3. Output spike time $z = \min\{t \mid P(t) \ge \theta\}$, or $\infty$ if no crossing occurs.

In training mode, `ExcitatoryNeuron` backpropagates through weights using a surrogate gradient (Gaussian window).

## `matterhorn_pytorch.tnn.compound_response`

Computes the piecewise-linear compound excitatory response $E_i(t, w_i)$ (Table 6.1, Figure 6.14).

```python
compound_response(
    t_rel: torch.Tensor,
    weight: torch.Tensor,
    tr: float,
    tf: float,
    gamma: float
) -> torch.Tensor
```

### Arguments

`t_rel (torch.Tensor)`: Relative time since input spike, arbitrary shape.

`weight (torch.Tensor)`: Synaptic weight $w_i$, broadcastable with `t_rel`.

`tr (float)`: Rise time $T_r$.

`tf (float)`: Fall time $T_f$.

`gamma (float)`: Scale factor $\gamma = T_{max}/W_{max}$.

### Returns

`response (torch.Tensor)`: Response amplitude, same shape as `t_rel`.

## `matterhorn_pytorch.tnn.membrane_potential`

Computes the membrane potential trajectory:

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

### Arguments

`times (torch.Tensor)`: Sample times, shape `[T]`.

`spike_times (torch.Tensor)`: Input spike times, shape `[B, I]`.

`weights (torch.Tensor)`: Synaptic weights, shape `[B, I]` or `[I]`.

`tr, tf, gamma (float)`: Response parameters.

`connectivity (torch.Tensor | None)`: Connection mask `[I]` or `[B, I]`; `0` means disconnected.

### Returns

`potential (torch.Tensor)`: Membrane potential $P(t)$, shape `[B, T]`.

## `matterhorn_pytorch.tnn.ExcitatoryNeuron`

Model II excitatory neuron. Input: first-spike timestamps per synapse (time volley). Output: neuron's first firing time.

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

### Constructor Arguments

`num_inputs (int)`: Number of input lines $M$.

`w_max (float)`: Maximum weight $W_{max}$.

`t_max (float)`: Time window $T_{max}$.

`tr (float)`: Rise time $T_r$.

`tf (float)`: Fall time $T_f$.

`theta (float)`: Firing threshold $\theta$.

`num_steps (int)`: Number of discrete samples along $P(t)$.

`normalize_input (bool)`: Whether to locally normalize the input volley.

`connectivity (torch.Tensor | None)`: Input connection mask `[M]`.

`device (torch.device)`: Compute device.

`dtype (torch.dtype)`: Data type.

### Trainable Parameters

`weight (nn.Parameter)`: Synaptic weights, shape `[M]`.

### Buffers

`connectivity (torch.Tensor)`: Connection mask `[M]`.

### Overridable Methods

#### `forward(self, spike_times: torch.Tensor, global_times: bool = True) -> Tuple[torch.Tensor, torch.Tensor]`

`spike_times (torch.Tensor)`: Input timestamps `[B, M]`; absent lines are `inf`.

`global_times (bool)`: If `True`, normalizes internally and denormalizes the output.

Returns:

- `out_time (torch.Tensor)`: Output spike time `[B]`;
- `t_min (torch.Tensor)`: Earliest input spike time `[B]`.

### Example Usage

```python
import torch
from matterhorn_pytorch.tnn import ExcitatoryNeuron

neuron = ExcitatoryNeuron(num_inputs=4, theta=5.0, num_steps=128)
neuron.weight.data.fill_(2.0)
neuron.eval()

spikes = torch.tensor([[0.5, 1.0, float("inf"), 2.0]])
out_time, t_min = neuron(spikes)
print(out_time)
```

### Supervised Training

Call `train()` and backpropagate through `weight`:

```python
neuron.train()
out, _ = neuron(spikes)
loss = out.sum()
loss.backward()
print(neuron.weight.grad)
```

Or use [`TNNTrainer`](./6_stdp_training.md) for end-to-end optimization.
