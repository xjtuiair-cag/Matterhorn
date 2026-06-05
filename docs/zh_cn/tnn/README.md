# `matterhorn_pytorch.tnn`

[回到 `matterhorn_pytorch`](../README.md)

[English](../../en_us/tnn/README.md)

[中文](../../zh_cn/tnn/README.md)

## `matterhorn_pytorch.tnn.functional` ： 时空代数的算子

该模块存储了 Temporal Neural Networks （TNNs）的基础——时空代数（Space-time Algebra）的 10 个运算符。除此之外，其还存储了一些 TNNs所用到的函数。

您可以阅读[文档](./1_functional.md)以了解更多函数的使用方法。

## `matterhorn_pytorch.tnn.component` ： TNN 组件

该模块含有 TNN 的基础组件，如双调排序网络（`Bitonic`）与发射单元（`Firing`）。这些类已在代码中实现，但未从 `tnn.__init__` 导出；请通过 `matterhorn_pytorch.tnn.component` 导入。

## 参考文献

[1] Smith J E, Martonosi M. Space-time computing with temporal neural networks[M]. Morgan & Claypool, 2017.