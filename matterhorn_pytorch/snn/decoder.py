# -*- coding: UTF-8 -*-
"""
脉冲神经网络的解码机制。
注意：此单元可能会改变张量形状。
"""


import torch
import torch.nn as nn
import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.skeleton import Module as _Module
from typing import Callable as _Callable, Optional as _Optional


class Decoder(_Module):
    def __init__(self) -> None:
        """
        解码器的基类。解码器是一个多时间步模型。
        """
        super().__init__()


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return super().extra_repr()


class SumSpike(Decoder):
    def __init__(self) -> None:
        """
        取张量在时间维度上的总值（总脉冲）。
        $$o_{i}=\sum_{t=1}^{T}{O_{i}^{K}(t)}$$
        """
        super().__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入张量，形状为[T,B,...]
        Returns:
            y (torch.Tensor): 输出张量，形状为[B,...]
        """
        y = _SF.decode_sum_spike(x)
        return y


class AverageSpike(Decoder):
    def __init__(self) -> None:
        """
        取张量在时间维度上的平均值（平均脉冲）。
        $$o_{i}=\frac{1}{T}\sum_{t=1}^{T}{O_{i}^{K}(t)}$$
        """
        super().__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入张量，形状为[T,B,...]
        Returns:
            y (torch.Tensor): 输出张量，形状为[B,...]
        """
        y = _SF.decode_avg_spike(x)
        return y


class TimeBased(Decoder):
    def __init__(self, empty_fill: float = -1, transform: _Optional[_Callable] = None) -> None:
        """
        基于时间的解码器。
        Args:
            empty_fill (float): 如果脉冲序列为全0序列，值应该用什么替代，在TNN中该参数应设为torch.inf
            transform (Callable): 将结果y如何变形
        """
        super().__init__()
        self.empty_fill = empty_fill
        self.transform = transform


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["empty_fill=%g" % self.empty_fill]) + ((", " + super().extra_repr()) if len(super().extra_repr()) else "")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入张量，形状为[T,B,...]
        Returns:
            y (torch.Tensor): 输出张量，形状为[B,...]
        """
        y = self.f_decode(x)
        if self.transform is not None:
            y = self.transform(y)
        return y


class MinTime(TimeBased):
    def __init__(self, empty_fill: float = -1, transform: _Optional[_Callable] = None) -> None:
        """
        取张量在时间维度上的最小值。
        $$o_{i}=min(tO_{i}^{K}(t))$$
        Args:
            empty_fill (float): 如果脉冲序列为全0序列，值应该用什么替代，在TNN中该参数应设为torch.inf
            transform (Callable): 将结果y如何变形
        """
        super().__init__(
            empty_fill = empty_fill,
            transform = transform
        )


    def f_decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        解码函数。
        Args:
            x (torch.Tensor): 输入张量，形状为[T,B,...]
        Returns:
            y (torch.Tensor): 输出张量，形状为[B,...]
        """
        y = _SF.decode_min_time(x, 0, self.empty_fill)
        return y


class AverageTime(TimeBased):
    def __init__(self, empty_fill: float = -1, transform: _Optional[_Callable] = None) -> None:
        """
        取张量在时间维度上的时间加权平均值
        $$o_{i}=\frac{1}{T}\sum_{t=1}^{T}{tO_{i}^{K}(t)}$$
        Args:
            empty_fill (float): 如果脉冲序列为全0序列，值应该用什么替代，在TNN中该参数应设为torch.inf
            transform (Callable): 将结果y如何变形
        """
        super().__init__(
            empty_fill = empty_fill,
            transform = transform
        )


    def f_decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        解码函数。
        Args:
            x (torch.Tensor): 输入张量，形状为[T,B,...]
        Returns:
            y (torch.Tensor): 输出张量，形状为[B,...]
        """
        y = _SF.decode_avg_time(x, 0, self.empty_fill)
        return y