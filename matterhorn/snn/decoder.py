# -*- coding: UTF-8 -*-
"""
脉冲神经网络的解码机制。
注意：此单元可能会改变张量形状。
"""


import torch
import torch.nn as nn
from matterhorn.snn.skeleton import Module
try:
    from rich import print
except:
    pass


class SumSpike(Module):
    def __init__(self) -> None:
        """
        取张量在时间维度上的总值（总脉冲）
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


class AverageSpike(Module):
    def __init__(self) -> None:
        """
        取张量在时间维度上的平均值（平均脉冲）
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


class MinTime(Module):
    def __init__(self, reset_after_process: bool = True) -> None:
        """
        取张量在时间维度上的时间加权平均值
        $$o_{i}=\frac{1}{T}\sum_{t=1}^{T}{tO_{i}^{K}(t)}$$
        @params:
            reset_after_process: bool 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__()
        self.current_time_step = 0
        self.reset_after_process = reset_after_process
        self.reset()


    def reset(self) -> None:
        """
        重置编码器
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


class AverageTime(Module):
    def __init__(self, reset_after_process: bool = True) -> None:
        """
        取张量在时间维度上的时间加权平均值
        $$o_{i}=\frac{1}{T}\sum_{t=1}^{T}{tO_{i}^{K}(t)}$$
        @params:
            reset_after_process: bool 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__()
        self.current_time_step = 0
        self.reset_after_process = reset_after_process
        self.time_mul = lambda x: x.permute(*torch.arange(x.ndim - 1, -1, -1))
        self.reset()


    def reset(self) -> None:
        """
        重置编码器
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
        mask = x.mean(dim = 0).le(0).to(x).detach().requires_grad_(True)
        y -= mask
        self.current_time_step += x.shape[0]
        if self.reset_after_process:
            self.reset()
        return y