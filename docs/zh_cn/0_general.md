# 基本介绍

## 1 `matterhorn_pytorch.data` ：神经形态数据集

该模块存储了常见的神经形态数据集处理方式，它可以将数据集（压缩包等形态）处理为 PyTorch 的张量。

## 2 `matterhorn_pytorch.lsm` ：液体状态机

该模块实现了液体状态机（LSM）。液体状态机的结构和功能可以参考文献 [1] 。

## 3 `matterhorn_pytorch.model` ：预定义模型

该模块定义了一些较为有名的 SNN 模型。

## 4 `matterhorn_pytorch.snn` ：脉冲神经网络

该模块是 Matterhorn 最主要的部分。该模块存储了脉冲神经网络（SNNs）的脉冲神经元、突触、容器、编码机制等常用组件，借此实现 SNNs 的部署。

您可以阅读[文档](./snn/0_general.md)以了解更多类与变量的使用方法。

## 5 `matterhorn_pytorch.tnn` ： Temporal Neural Networks

该模块实现了 Temporal Neural Networks （TNNs）。 Temporal Neural Networks 的结构和功能可以参考文献 [2] 。

## 6 `matterhorn_pytorch.training` ：基于脉冲的训练

除反向传播之外，脉冲神经元有专属的训练方式。该模块定义了脉冲神经元的训练方式，如脉冲时序依赖可塑性（STDP）等。

## 7 `matterhorn_pytorch.util` ：小工具

该模块携带有对研究和部署基于脉冲的神经网络具备用途的一些小工具，如画图等。

## 参考文献

[1] Maass W. Liquid state machines: motivation, theory, and applications[J]. Computability in context: computation and logic in the real world, 2011: 275-296.

[2] Smith J E, Martonosi M. Space-time computing with temporal neural networks[M]. Morgan & Claypool, 2017.