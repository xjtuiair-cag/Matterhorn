# `matterhorn_pytorch.tnn`

[Back to `matterhorn_pytorch`](../README.md)

[English](../../en_us/tnn/README.md)

[中文](../../zh_cn/tnn/README.md)

## `matterhorn_pytorch.tnn.functional`: Space-Time Algebra Operators

This module stores the basic operators of space-time algebra used by Temporal Neural Networks (TNNs), including comparison operators, conversion between time and spike representations, and competitive-logic (RL) operators.

You can read [the documentation](./1_functional.md) to learn more about the usage of functions.

## `matterhorn_pytorch.tnn.component`: TNN Building Blocks

This module contains TNN components such as the bitonic sorting network (`Bitonic`) and firing unit (`Firing`). These classes are defined in code but not exported from `tnn.__init__`; import them via `matterhorn_pytorch.tnn.component`.

## References

[1] Smith J E, Martonosi M. Space-time computing with temporal neural networks[M]. Morgan & Claypool, 2017.