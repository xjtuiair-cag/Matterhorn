# `matterhorn_pytorch.snn`

[Back to `matterhorn_pytorch`](../0_general.md)

[English](../../en_us/snn/0_general.md)

[中文](../../zh_cn/snn/0_general.md)

## `matterhorn_pytorch.snn.container`: Container for SNN Modules

This module covers containers that can encapsulate SNN modules. Through these containers, SNN modules based on `matterhorn_pytorch.snn.Module` can form spatio-temporal computing networks.

Containing `matterhorn_pytorch.snn` modules: `Spatial`, `Temporal`, and `Sequential`.

You can read the documentation to learn more about the usage of classes and variables.

## `matterhorn_pytorch.snn.decoder`: Decoder for SNNs

This module contains decoders that transform spike sequences into analog values by statistically analyzing information along the temporal dimension. It serves as a bridge for the output of SNNs to be used as input for ANNs.

Containing `matterhorn_pytorch.snn` modules: `SumSpikeDecoder`, `AvgSpikeDecoder`, `MinTimeDecoder`, and `AvgTimeDecoder`.

You can read the documentation to learn more about the usage of classes and variables.

## `matterhorn_pytorch.snn.encoder`: Encoder for SNNs

This module contains encoders that encode analog values along the temporal dimension into spike sequences. It serves as a bridge for the output of ANNs to be used as input for SNNs.

Containing `matterhorn_pytorch.snn` modules: `DirectEncoder`, `PoissonEncoder`, `SoftMaxEncoder`, and `TemporalEncoder`.

You can read the documentation to learn more about the usage of classes and variables.

## `matterhorn_pytorch.snn.functional`: Related Functions for SNNs

This module defines some common functions in SNNs, such as the Heaviside step function, etc.

You can read the documentation to learn more about the usage of functions.

## `matterhorn_pytorch.snn.layer`: Pulse-to-Pulse Operations for SNNs

The units in this module, both input and output are spike sequences. It contains SRM0 neurons, pooling layers, etc.

Containing `matterhorn_pytorch.snn` modules: `SRM0Linear`, `STDPLinear`, `MaxPool1d`, `MaxPool2d`, `MaxPool3d`, `AvgPool1d`, `AvgPool2d`, `AvgPool3d`, `Flatten`, and `Unflatten`.

You can read the documentation to learn more about the usage of classes and variables.

## `matterhorn_pytorch.snn.skeleton`: Base Class for SNN Modules

A base class inherited by all modules in `matterhorn_pytorch.snn`, serving as the skeleton of `matterhorn_pytorch.snn`.

Containing `matterhorn_pytorch.snn` modules: `Module`.

You can read the documentation to learn more about the usage of classes and variables.

## `matterhorn_pytorch.snn.soma`: Soma of Spiking Neurons

This module defines the soma of SNNs, where the input is the membrane potential (analog value), and the output is a spike sequence. It contains IF, LIF neurons, etc.

Containing `matterhorn_pytorch.snn` modules: `IF`, `LIF`, `QIF`, `ExpIF`, `Izhikevich`, `KLIF`, and `LIAF`.

You can read the documentation to learn more about the usage of classes and variables.

## `matterhorn_pytorch.snn.surrogate`: Surrogate Gradients for the Heaviside Step Function of Spiking Neurons

This module defines surrogate gradients for the Heaviside step function of spiking neurons. The specific definition can be found in reference [1].

Containing `matterhorn_pytorch.snn` modules: `Rectangular`, `Polynomial`, `Sigmoid`, and `Gaussian`.

You can read the documentation to learn more about the usage of classes and variables.

## `matterhorn_pytorch.snn.synapse`: Synapse of Spiking Neurons

This module defines the synapse of SNNs, where the input is a spike sequence and the output is the membrane potential (analog value). It contains operations such as fully connected layers, convolutions, etc.

Containing `matterhorn_pytorch.snn` modules: `Linear`, `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`, `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`, and `LayerNorm`.

You can read the documentation to learn more about the usage of classes and variables.

## References

[1] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.