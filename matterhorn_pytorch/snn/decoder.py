# -*- coding: UTF-8 -*-
"""
脉冲神经网络的解码机制。
注意：此单元可能会改变张量形状。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.__func__ import transpose as _transpose
from matterhorn_pytorch.snn.skeleton import Module as _Module
from typing import Callable as _Callable


class Decoder(_Module):
    def __init__(self, reset_after_process: bool = False) -> None:
        """
        解码器的基类。解码器是一个多时间步模型。
        Args:
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__(
            multi_time_step = True,
            reset_after_process = reset_after_process
        )


    @property
    def supports_single_time_step(self) -> bool:
        """
        是否支持单个时间步。
        Returns:
            if_support (bool): 是否支持单个时间步
        """
        return False


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
        y = x.sum(dim = 0)
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
        y = x.mean(dim = 0)
        return y


class TimeBased(Decoder):
    def __init__(self, empty_fill: float = -1, transform: _Callable = lambda x: x, reset_after_process: bool = True) -> None:
        """
        基于时间的解码器。
        Args:
            empty_fill (float): 如果脉冲序列为全0序列，值应该用什么替代，在TNN中该参数应设为torch.inf
            transform (Callable): 将结果y如何变形
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__(
            reset_after_process = reset_after_process
        )
        self.empty_fill = empty_fill
        self.transform = transform
        self.reset()


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return "empty_fill=%g, reset_after_process=%s" % (self.empty_fill, str(self.reset_after_process))


    def reset(self) -> _Module:
        """
        重置编码器。
        """
        self.current_time_step = 0
        return super().reset()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入张量，形状为[T,B,...]
        Returns:
            y (torch.Tensor): 输出张量，形状为[B,...]
        """
        y = self.f_decode(x).to(x)
        y = y.to(x)
        mask = x.mean(dim = 0) > 0
        y = torch.where(mask, y, torch.full_like(y, self.empty_fill))
        self.current_time_step += x.shape[0]
        y = self.transform(y)
        return y


class MinTime(TimeBased):
    def __init__(self, empty_fill: float = -1, transform: _Callable = lambda x: x, reset_after_process: bool = True) -> None:
        """
        取张量在时间维度上的最小值。
        $$o_{i}=min(tO_{i}^{K}(t))$$
        Args:
            empty_fill (float): 如果脉冲序列为全0序列，值应该用什么替代，在TNN中该参数应设为torch.inf
            transform (Callable): 将结果y如何变形
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__(
            empty_fill = empty_fill,
            transform = transform,
            reset_after_process = reset_after_process
        )


    def f_decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        解码函数。
        Args:
            x (torch.Tensor): 输入张量，形状为[T,B,...]
        Returns:
            y (torch.Tensor): 输出张量，形状为[B,...]
        """
        y = torch.argmax(x, dim = 0) + self.current_time_step
        return y


class AverageTime(TimeBased):
    def __init__(self, empty_fill: float = -1, transform: _Callable = lambda x: x, reset_after_process: bool = True) -> None:
        """
        取张量在时间维度上的时间加权平均值
        $$o_{i}=\frac{1}{T}\sum_{t=1}^{T}{tO_{i}^{K}(t)}$$
        Args:
            empty_fill (float): 如果脉冲序列为全0序列，值应该用什么替代，在TNN中该参数应设为torch.inf
            transform (Callable): 将结果y如何变形
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__(
            empty_fill = empty_fill,
            transform = transform,
            reset_after_process = reset_after_process
        )


    def f_decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        解码函数。
        Args:
            x (torch.Tensor): 输入张量，形状为[T,B,...]
        Returns:
            y (torch.Tensor): 输出张量，形状为[B,...]
        """
        xt = _transpose(_transpose(x) * torch.arange(x.shape[0]).to(x))
        tsum = torch.sum(xt, dim = 0)
        xsum = torch.sum(x, dim = 0)
        mask = xsum > 0
        y = torch.where(mask, tsum / xsum, torch.zeros_like(xsum))
        return y