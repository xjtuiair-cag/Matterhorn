import torch
import torch.nn as nn


"""
脉冲神经网络的解码机制
注意：此单元可能会改变张量形状
"""


class Sum(nn.Module):
    def __init__(self) -> None:
        """
        取张量在时间维度上的总值（总脉冲）
        """
        super().__init__()
    

    def forward(self, x: torch.Tensor):
        """
        前向传播函数
        @params:
            x: torch.Tensor 输入张量，形状为[T,B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[B,...]
        """
        y = x.sum(dim = 0)
        return y


class Average(nn.Module):
    def __init__(self) -> None:
        """
        取张量在时间维度上的平均值（平均脉冲）
        """
        super().__init__()
    

    def forward(self, x: torch.Tensor):
        """
        前向传播函数
        @params:
            x: torch.Tensor 输入张量，形状为[T,B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[B,...]
        """
        y = x.mean(dim = 0)
        return y