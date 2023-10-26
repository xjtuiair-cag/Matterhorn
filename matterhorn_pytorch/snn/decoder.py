# -*- coding: UTF-8 -*-
"""
脉冲神经网络的解码机制。
注意：此单元可能会改变张量形状。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.snn.skeleton import Module
try:
    from rich import print
except:
    pass


class Decoder(Module):
    def __init__(self, reset_after_process: bool = False) -> None:
        """
        解码器的基类。解码器是一个多时间步模型。
        @params:
            reset_after_process: bool 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__(
            multi_time_step = True,
            reset_after_process = reset_after_process
        )


    def supports_single_time_step(self) -> bool:
        """
        是否支持单个时间步。
        @return:
            if_support: bool 是否支持单个时间步
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
        @params:
            x: torch.Tensor 输入张量，形状为[T,B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[B,...]
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
        @params:
            x: torch.Tensor 输入张量，形状为[T,B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[B,...]
        """
        y = x.mean(dim = 0)
        return y


class MinTime(Decoder):
    def __init__(self, reset_after_process: bool = True) -> None:
        """
        取张量在时间维度上的时间加权平均值。
        $$o_{i}=\frac{1}{T}\sum_{t=1}^{T}{tO_{i}^{K}(t)}$$
        @params:
            reset_after_process: bool 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__(
            reset_after_process = True
        )
        self.current_time_step = 0
        self.reset()


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return "reset_after_process=%s" % (str(self.reset_after_process),)


    def reset(self) -> None:
        """
        重置编码器。
        """
        self.current_time_step = 0
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            x: torch.Tensor 输入张量，形状为[T,B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[B,...]
        """
        y = torch.argmax(x, dim = 0) + self.current_time_step
        mask = x.mean(dim = 0).le(0).to(x).detach().requires_grad_(True)
        y -= mask
        self.current_time_step += x.shape[0]
        if self.reset_after_process:
            self.reset()
        return y


class AverageTime(Decoder):
    def __init__(self, reset_after_process: bool = True) -> None:
        """
        取张量在时间维度上的时间加权平均值
        $$o_{i}=\frac{1}{T}\sum_{t=1}^{T}{tO_{i}^{K}(t)}$$
        @params:
            reset_after_process: bool 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__(
            reset_after_process = reset_after_process
        )
        self.current_time_step = 0
        self.time_mul = lambda x: x.permute(*torch.arange(x.ndim - 1, -1, -1))
        self.reset()


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return "reset_after_process=%s" % (str(self.reset_after_process),)


    def reset(self) -> None:
        """
        重置编码器。
        """
        self.current_time_step = 0
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            x: torch.Tensor 输入张量，形状为[T,B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[B,...]
        """
        t = torch.arange(x.shape[0]).to(x)
        xT = self.time_mul(x)
        xTt = xT * t
        xTt /= x.shape[0]
        y = self.time_mul(xTt) + self.current_time_step
        y = y.mean(dim = 0)
        mask = x.mean(dim = 0).le(0).to(x).detach().requires_grad_(True)
        y -= mask
        self.current_time_step += x.shape[0]
        if self.reset_after_process:
            self.reset()
        return y