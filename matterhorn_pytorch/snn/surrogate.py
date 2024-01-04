# -*- coding: UTF-8 -*-
"""
阶跃函数及其替代导数。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.snn.functional import heaviside_rectangular, heaviside_polynomial, heaviside_sigmoid, heaviside_gaussian
try:
    from rich import print
except:
    pass


class Rectangular(nn.Module):
    def __init__(self, a: float = 2.0) -> None:
        """
        Heaviside阶跃函数，替代梯度为矩形函数
        Args:
            a (float): 参数a，决定矩形函数的形状
        """
        super().__init__()
        self.func = heaviside_rectangular()
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
        return self.func.apply(x, self.a)


class Polynomial(nn.Module):
    def __init__(self, a: float = 1.0) -> None:
        """
        Heaviside阶跃函数，替代梯度为多项式函数
        Args:
            a (float): 参数a，决定多项式函数的形状
        """
        super().__init__()
        self.func = heaviside_polynomial()
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
        return self.func.apply(x, self.a)


class Sigmoid(nn.Module):
    def __init__(self, a: float = 1.0) -> None:
        """
        Heaviside阶跃函数，替代梯度为Sigmoid函数
        Args:
            a (float): 参数a，决定Sigmoid函数的形状
        """
        super().__init__()
        self.func = heaviside_sigmoid()
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
        return self.func.apply(x, self.a)


class Gaussian(nn.Module):
    def __init__(self, a: float = 1.0) -> None:
        """
        Heaviside阶跃函数，替代梯度为高斯函数
        Args:
            a (float): 参数a，决定高斯函数的形状
        """
        super().__init__()
        self.func = heaviside_gaussian()
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
        return self.func.apply(x, self.a)