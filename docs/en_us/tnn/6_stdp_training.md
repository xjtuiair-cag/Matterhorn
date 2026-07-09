# `matterhorn_pytorch.tnn.stdp` and `matterhorn_pytorch.tnn.training`

[Back to `matterhorn_pytorch.tnn`](./README.md)

[English](../../en_us/tnn/6_stdp_training.md)

[中文](../../zh_cn/tnn/6_stdp_training.md)

## Module Introduction

These modules provide **training** for TNNs, corresponding to Smith's learning mechanisms:

1. **STDP** (Section 6.4.3): local unsupervised weight updates from input/output spike timing;
2. **Model II mean method** (Section 6.6.2): $w_i = (T_{max} - \bar{t}_i) / \gamma$;
3. **Supervised learning**: optimize weights via surrogate-gradient backprop on output spike times.

## `matterhorn_pytorch.tnn.stdp_gate`

Piecewise-linear STDP gate $G(\Delta t)$ with $\Delta t = t_{in} - t_{out}$:

- Input before output ($\Delta t < 0$) → potentiation ($+1$);
- Input after output ($\Delta t > 0$) → depression ($-1$).

```python
stdp_gate(
    delta_t: torch.Tensor,
    t_max: float,
    slope: float = 0.1
) -> torch.Tensor
```

### Arguments

`delta_t (torch.Tensor)`: Time difference $\Delta t$.

`t_max (float)`: Encoding window $T_{max}$ for far-end saturation.

`slope (float)`: Linear slope near $\Delta t = 0$.

### Returns

`gate (torch.Tensor)`: Update direction and magnitude.

## `matterhorn_pytorch.tnn.volley_stdp_update`

Local STDP weight update from volleys (Smith Section 6.4.3).

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

### Arguments

`weights (torch.Tensor)`: Synaptic weights `[I]` or `[O, I]`.

`input_times (torch.Tensor)`: Input spike times `[B, I]` or `[I]` (normalized frame).

`output_time (torch.Tensor)`: Output spike time scalar or `[B]`.

`learning_rate (float)`: Learning rate $\lambda$.

`t_max (float)`: $T_{max}$.

`w_max (float)`: $W_{max}$.

`connectivity (torch.Tensor | None)`: Connection mask.

### Returns

`new_weights (torch.Tensor)`: Updated weights clipped to $[0, W_{max}]$.

## `matterhorn_pytorch.tnn.mean_time_to_weight`

Model II non-STDP training:

$$w_i = \frac{T_{max} - \bar{t}_i}{\gamma}$$

```python
mean_time_to_weight(
    mean_times: torch.Tensor,
    t_max: float,
    gamma: float
) -> torch.Tensor
```

### Arguments

`mean_times (torch.Tensor)`: Mean spike time per input line.

`t_max (float)`: $T_{max}$.

`gamma (float)`: $\gamma = T_{max}/W_{max}$.

### Returns

`weights (torch.Tensor)`: Trained weights.

## `matterhorn_pytorch.tnn.spike_time_mse_loss`

Mean squared error on output spike times. Absent spikes are replaced with a large penalty value.

```python
spike_time_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fill: float = 1e6
) -> torch.Tensor
```

### Arguments

`pred (torch.Tensor)`: Predicted times `[B]` or `[B, N]`.

`target (torch.Tensor)`: Target times.

`fill (float)`: Substitute value for absent spikes.

### Returns

`loss (torch.Tensor)`: Scalar loss.

## `matterhorn_pytorch.tnn.train_excitatory_neuron_stdp`

Unsupervised STDP training for a single excitatory neuron (in-place weight update).

```python
train_excitatory_neuron_stdp(
    neuron: ExcitatoryNeuron,
    training_volleys: Iterable[torch.Tensor],
    learning_rate: float = 0.05,
    epochs: int = 1
) -> ExcitatoryNeuron
```

### Arguments

`neuron (ExcitatoryNeuron)`: Neuron to train.

`training_volleys (Iterable[torch.Tensor])`: Training volleys `[B, I]`.

`learning_rate (float)`: Learning rate.

`epochs (int)`: Number of passes over the training set.

### Returns

Trained neuron (modified in place).

## `matterhorn_pytorch.tnn.train_column_mean_weights`

Sets weights for each neuron in an excitatory column using the Model II mean method.

```python
train_column_mean_weights(
    column: ExcitatoryColumn,
    training_volleys: torch.Tensor
) -> ExcitatoryColumn
```

### Arguments

`column (ExcitatoryColumn)`: Excitatory column.

`training_volleys (torch.Tensor)`: Training volleys `[S, I]`.

### Returns

Trained column (modified in place).

## `matterhorn_pytorch.tnn.TNNTrainer`

Supervised trainer using Adam for models containing `ExcitatoryNeuron`, `ExcitatoryColumn`, or `CognitiveColumn`.

```python
TNNTrainer(model: nn.Module, learning_rate: float = 1e-2)
```

### Constructor Arguments

`model (nn.Module)`: TNN model to train.

`learning_rate (float)`: Adam learning rate.

### Methods

#### `supervised_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float`

`inputs (torch.Tensor)`: Input volley `[B, I]`.

`targets (torch.Tensor)`: Target output times `[B]` or `[B, N]`.

Returns: Loss value for this step (`float`).

### Example Usage

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
