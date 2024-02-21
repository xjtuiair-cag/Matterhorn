# -*- coding: UTF-8 -*-
"""
阶跃函数及其替代导数。
"""


import torch
import torch.nn as nn
import matterhorn_pytorch.snn.functional as F
from matterhorn_pytorch.snn.skeleton import Module
try:
    from rich import print
except:
    pass


class Rectangular(Module):
    def __init__(self, a: float = 1.0) -> None:
        """
        Heaviside阶跃函数，替代梯度为矩形函数
        Args:
            a (float): 参数a，决定矩形函数的形状
        """
        super().__init__()
        self.a = a
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return "a=%g" % (self.a)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            o (torch.Tensor): 输出张量
        """
        return F.heaviside_rectangular(x, self.a)


class Polynomial(Module):
    def __init__(self, a: float = 4.0) -> None:
        """
        Heaviside阶跃函数，替代梯度为多项式函数
        Args:
            a (float): 参数a，决定多项式函数的形状
        """
        super().__init__()
        self.a = a
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return "a=%g" % (self.a)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            o (torch.Tensor): 输出张量
        """
        return F.heaviside_polynomial(x, self.a)


class Sigmoid(Module):
    def __init__(self, a: float = 0.25) -> None:
        """
        Heaviside阶跃函数，替代梯度为Sigmoid函数
        Args:
            a (float): 参数a，决定Sigmoid函数的形状
        """
        super().__init__()
        self.a = a
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return "a=%g" % (self.a)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            o (torch.Tensor): 输出张量
        """
        return F.heaviside_sigmoid(x, self.a)


class Gaussian(Module):
    def __init__(self, a: float = 0.16) -> None:
        """
        Heaviside阶跃函数，替代梯度为高斯函数
        Args:
            a (float): 参数a，决定高斯函数的形状
        """
        super().__init__()
        self.a = a
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return "a=%g" % (self.a)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            o (torch.Tensor): 输出张量
        """
        return F.heaviside_gaussian(x, self.a)