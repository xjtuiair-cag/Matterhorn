# `matterhorn_pytorch`

[Back to Main Document](../../README.md)

[English](../en_us/README.md)

[中文](../zh_cn/README.md)

## `matterhorn_pytorch.data`: Neuromorphic Dataset

This module stores common methods for handling neural morphology datasets, which can process datasets (in the form of compressed files, etc.) into PyTorch tensors.

You can read [the documentation](./data/README.md) to learn more about the usage of classes and variables.

## `matterhorn_pytorch.snn`: Spiking Neural Networks

This module is the main part of Matterhorn. It stores commonly used components of Spiking Neural Networks (SNNs), such as spiking neurons, synapses, containers, encoding mechanisms, etc., to implement the deployment of SNNs.

You can read [the documentation](./snn/README.md) to learn more about the usage of classes and variables.

## `matterhorn_pytorch.tnn`: Temporal Neural Networks

This module implements Temporal Neural Networks (TNNs). The structure and functionality of Temporal Neural Networks can be referred to in reference [2].

You can read [the documentation](./tnn/README.md) to learn more about the usage of classes and variables.

## `matterhorn_pytorch.training`: Spike-Based Training

In addition to backpropagation, spiking neurons have their own training methods. This module defines the training methods of spiking neurons, such as Spike Timing-Dependent Plasticity (STDP), etc.

You can read [the documentation](./training/README.md) to learn more about the usage of classes and variables.

## `matterhorn_pytorch.util`: Utilities

This module carries some utilities useful for researching and deploying spike-based neural networks, such as plotting tools, etc.

You can read [the documentation](./util/README.md) to learn more about the usage of classes and variables.

## References

[1] Maass W. Liquid state machines: motivation, theory, and applications[J]. Computability in context: computation and logic in the real world, 2011: 275-296.

[2] Smith J E, Martonosi M. Space-time computing with temporal neural networks[M]. Morgan & Claypool, 2017.