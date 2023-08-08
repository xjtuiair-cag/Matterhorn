import torch
import torch.nn as nn


"""
脉冲神经网络的解码机制。
注意：此单元可能会改变张量形状。
"""


class SumSpike(nn.Module):
    def __init__(self) -> None:
        """
        取张量在时间维度上的总值（总脉冲）
        $$o_{i}=\sum_{t=1}^{T}{O_{i}^{K}(t)}$$
        """
        super().__init__()
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        @params:
            x: torch.Tensor 输入张量，形状为[T,B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[B,...]
        """
        y = x.sum(dim = 0)
        return y


class AverageSpike(nn.Module):
    def __init__(self) -> None:
        """
        取张量在时间维度上的平均值（平均脉冲）
        $$o_{i}=\frac{1}{T}\sum_{t=1}^{T}{O_{i}^{K}(t)}$$
        """
        super().__init__()
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        @params:
            x: torch.Tensor 输入张量，形状为[T,B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[B,...]
        """
        y = x.mean(dim = 0)
        return y


class MinTime(nn.Module):
    def __init__(self, time_steps: int = 1, reset_after_process: bool = True) -> None:
        """
        取张量在时间维度上的时间加权平均值
        $$o_{i}=\frac{1}{T}\sum_{t=1}^{T}{tO_{i}^{K}(t)}$$
        @params:
            time_steps: int 执行的时间步长
            reset_after_process: bool 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__()
        self.time_steps = time_steps
        self.reset_after_process = reset_after_process
        self.n_reset()


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来
        @return:
            repr_str: str 参数表
        """
        return "time_steps=%d" % (self.time_steps,)


    def n_reset(self) -> None:
        """
        重置编码器
        """
        self.current_time_step = 0
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        @params:
            x: torch.Tensor 输入张量，形状为[T,B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[B,...]
        """
        y = x.mean(dim = 0)
        # TODO: 前向传播具体算法
        if self.reset_after_process:
            self.n_reset()
        return y


class AverageTime(nn.Module):
    def __init__(self, time_steps: int = 1, reset_after_process: bool = True) -> None:
        """
        取张量在时间维度上的时间加权平均值
        $$o_{i}=\frac{1}{T}\sum_{t=1}^{T}{tO_{i}^{K}(t)}$$
        @params:
            time_steps: int 执行的时间步长
            reset_after_process: bool 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__()
        self.time_steps = time_steps
        self.reset_after_process = reset_after_process
        self.n_reset()


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来
        @return:
            repr_str: str 参数表
        """
        return "time_steps=%d" % (self.time_steps,)


    def n_reset(self) -> None:
        """
        重置编码器
        """
        self.current_time_step = 0
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        @params:
            x: torch.Tensor 输入张量，形状为[T,B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[B,...]
        """
        y = x.mean(dim = 0)
        # TODO: 前向传播具体算法
        if self.reset_after_process:
            self.n_reset()
        return y