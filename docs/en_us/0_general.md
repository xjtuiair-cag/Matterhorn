# General Introduction

## 1 `matterhorn_pytorch.data`: Neuromorphic Dataset

This module stores common methods for handling neural morphology datasets, which can process datasets (in the form of compressed files, etc.) into PyTorch tensors.

You can read the documentation to learn more about the usage of classes and variables.

## 2 `matterhorn_pytorch.lsm`: Liquid State Machines

This module implements the Liquid State Machines (LSMs). The structure and functionality of the Liquid State Machines can be referred to in reference [1].

You can read the documentation to learn more about the usage of classes and variables.

## 3 `matterhorn_pytorch.model`: Predefined Models

This module defines some well-known SNN models.

You can read the documentation to learn more about the usage of classes and variables.

## 4 `matterhorn_pytorch.snn`: Spiking Neural Networks

This module is the main part of Matterhorn. It stores commonly used components of Spiking Neural Networks (SNNs), such as spiking neurons, synapses, containers, encoding mechanisms, etc., to implement the deployment of SNNs.

You can read [the documentation](./snn/0_general.md) to learn more about the usage of classes and variables.

## 5 `matterhorn_pytorch.tnn`: Temporal Neural Networks

This module implements Temporal Neural Networks (TNNs). The structure and functionality of Temporal Neural Networks can be referred to in reference [2].

You can read the documentation to learn more about the usage of classes and variables.

## 6 `matterhorn_pytorch.training`: Spike-Based Training

In addition to backpropagation, spiking neurons have their own training methods. This module defines the training methods of spiking neurons, such as Spike Timing-Dependent Plasticity (STDP), etc.

You can read the documentation to learn more about the usage of classes and variables.

## 7 `matterhorn_pytorch.util`: Utilities

This module carries some utilities useful for researching and deploying spike-based neural networks, such as plotting tools, etc.

You can read the documentation to learn more about the usage of classes and variables.

## References

[1] Maass W. Liquid state machines: motivation, theory, and applications[J]. Computability in context: computation and logic in the real world, 2011: 275-296.

[2] Smith J E, Martonosi M. Space-time computing with temporal neural networks[M]. Morgan & Claypool, 2017.