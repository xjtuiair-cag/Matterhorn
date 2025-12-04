# `matterhorn_pytorch.snn`

[回到 `matterhorn_pytorch`](../README.md)

[English](../../en_us/snn/README.md)

[中文](../../zh_cn/snn/README.md)

## `matterhorn_pytorch.snn.container` ： SNN 模块的容器

该模块涵盖了能够包裹 SNN 模块的容器，通过这些容器，基于`matterhorn_pytorch.snn.Module`的 SNN 模块可以组成时空计算网络。

含有的 `matterhorn_pytorch.snn` 模块： `Sequential` 。

您可以阅读[文档](./7_container.md)以了解更多类与变量的使用方法。

## `matterhorn_pytorch.snn.decoder` ： SNN 的解码机制

该模块中含有通过统计时间维度上的信息，将脉冲序列转化为模拟值的解码器。是 SNN 的输出作为 ANN 的输入的桥梁。

含有的 `matterhorn_pytorch.snn` 模块： `SumSpikeDecoder` 、 `AvgSpikeDecoder` 、 `MinTimeDecoder` 和 `AvgTimeDecoder` 。

您可以阅读[文档](./9_decoder.md)以了解更多类与变量的使用方法。

## `matterhorn_pytorch.snn.encoder` ： SNN 的编码机制

该模块中含有将模拟值在时间维度上编码，转化为脉冲序列的编码器。是 ANN 的输出作为 SNN 的输入的桥梁。

含有的 `matterhorn_pytorch.snn` 模块： `DirectEncoder` 、 `PoissonEncoder` 和 `TemporalEncoder` 。

您可以阅读[文档](./8_encoder.md)以了解更多类与变量的使用方法。

## `matterhorn_pytorch.snn.functional` ： SNN 的相关函数

该模块中定义了一些 SNNs 中常见的函数，如 Heaviside 阶跃函数等。

您可以阅读[文档](./1_functional.md)以了解更多函数的使用方法。

## `matterhorn_pytorch.snn.layer` ： SNN 的脉冲对脉冲操作

该模块中的单元，输入和输出皆为脉冲序列。含有展开层、池化层等。

含有的 `matterhorn_pytorch.snn` 模块： `STDPLinear` 、 `MaxPool1d` 、 `MaxPool2d` 、 `MaxPool3d` 、 `AvgPool1d` 、 `AvgPool2d` 、 `AvgPool3d` 、 `Flatten` 、 `Unflatten` 、 `Dropout` 、 `Dropout1d` 、 `Dropout2d` 和 `Dropout3d` 。

您可以阅读[文档](./6_layer.md)以了解更多类与变量的使用方法。

## `matterhorn_pytorch.snn.skeleton` ： SNN 模块的基类

所有 `matterhorn_pytorch.snn` 中的模块共同继承的一个基类， `matterhorn_pytorch.snn` 的骨架。

含有的 `matterhorn_pytorch.snn` 模块： `Module` 。

您可以阅读[文档](./2_skeleton.md)以了解更多类与变量的使用方法。

## `matterhorn_pytorch.snn.soma` ： 脉冲神经元的胞体

该模块定义了 SNNs 的脉冲神经元胞体，它们的输入为输入电位（模拟值），输出为脉冲序列。含有 IF 、 LIF 神经元等。

含有的 `matterhorn_pytorch.snn` 模块： `IF` 、 `LIF` 、 `QIF` 、 `ExpIF` 、 `Izhikevich` 、 `KLIF` 和 `LIAF` 。

您可以阅读[文档](./4_soma.md)以了解更多类与变量的使用方法。

## `matterhorn_pytorch.snn.firing` ： 脉冲神经元 Heaviside 阶跃函数的替代梯度

该模块定义了脉冲神经元 Heaviside 阶跃函数的替代梯度，具体的定义您可以在参考文献 [1] 中找到。

含有的 `matterhorn_pytorch.snn` 模块： `Rectangular` 、 `Polynomial` 、 `Sigmoid` 和 `Gaussian` 。

您可以阅读[文档](./3_firing.md)以了解更多类与变量的使用方法。

## `matterhorn_pytorch.snn.synapse` ： 脉冲神经元的突触

该模块定义了 SNNs 的脉冲神经元突触，它们的输入为脉冲序列，输出为输入电位（模拟值）。含有全连接、卷积操作等。

含有的 `matterhorn_pytorch.snn` 模块： `Linear` 、 `Conv1d` 、 `Conv2d` 、 `Conv3d` 、 `ConvTranspose1d` 、 `ConvTranspose2d` 、 `ConvTranspose3d` 、 `BatchNorm1d` 、 `BatchNorm2d` 、 `BatchNorm3d` 和 `LayerNorm` 。

您可以阅读[文档](./5_synapse.md)以了解更多类与变量的使用方法。

## 参考文献

[1] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.