# `matterhorn_pytorch.tnn`

[Back to `matterhorn_pytorch`](../README.md)

[English](../../en_us/tnn/README.md)

[中文](../../zh_cn/tnn/README.md)

## `matterhorn_pytorch.tnn.functional`: Space-Time Algebra Operators

This module stores the basic operators of space-time algebra used by Temporal Neural Networks (TNNs), including comparison operators, conversion between time and spike representations, and competitive-logic (RL) operators.

Read [the documentation](./1_functional.md) for details.

## `matterhorn_pytorch.tnn.volley`: Volley Encoding and Decoding

Utilities for normalizing time volleys, value↔time conversion, OnOff encoding, and bundle merge—the foundational representation for TNN inputs and outputs.

Read [the documentation](./2_volley.md).

## `matterhorn_pytorch.tnn.response` and `matterhorn_pytorch.tnn.neuron`: Excitatory Neurons

Implements Smith Model II piecewise-linear compound responses, membrane-potential summation, and the backpropagation-capable `ExcitatoryNeuron`.

Read [the documentation](./3_neuron.md).

## `matterhorn_pytorch.tnn.inhibition`: Inhibitory Columns

Feedforward inhibition (FFI) and lateral inhibition (LI / tk-WTA) used inside cognitive columns.

Read [the documentation](./4_inhibition.md).

## `matterhorn_pytorch.tnn.column`, `cognitive`, `container`: Columns and Network Containers

- `ExcitatoryColumn`: parallel excitatory neuron array (EC);
- `CognitiveColumn`: FFI → EC → LI cognitive column (CC);
- `Sequential`: multi-layer TNN chaining container.

Read [the documentation](./5_column_cognitive.md).

## `matterhorn_pytorch.tnn.stdp` and `matterhorn_pytorch.tnn.training`: Learning and Training

STDP unsupervised learning, Model II mean-weight initialization, and supervised `TNNTrainer` based on output spike times.

Read [the documentation](./6_stdp_training.md).

## `matterhorn_pytorch.tnn.component`: TNN Building Blocks

Contains components such as the bitonic sorting network (`Bitonic`) and firing unit (`Firing`). Import via `from matterhorn_pytorch.tnn import Bitonic, Firing` or `matterhorn_pytorch.tnn.component`.

## Quick Start

```python
import torch
from matterhorn_pytorch.tnn import CognitiveColumn, Sequential, TNNTrainer

cc = CognitiveColumn(num_inputs=8, num_neurons=4, seed=42)
spikes = torch.tensor([[1.0, 1.2, 1.5, 2.0,
                        float("inf"), float("inf"), float("inf"), float("inf")]])
out, states = cc(merged_input=spikes)

net = Sequential(cc)
trainer = TNNTrainer(net)
```

## References

[1] Smith J E, Martonosi M. Space-time computing with temporal neural networks[M]. Morgan & Claypool, 2017.
