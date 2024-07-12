# -*- coding: UTF-8 -*-
"""
阶跃函数及其替代导数。
"""


import torch
import torch.nn as nn
import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.skeleton import Module as _Module


class Firing(_Module):
    def __init__(self, multi_spikes: bool = False):
        """
        发射函数，产生脉冲的函数。
        """
        super().__init__()
        self.multi_spikes = multi_spikes


class Rectangular(Firing):
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
        return "a=%g" % self.a
    

    def forward(self, u: torch.Tensor, u_threshold: torch.Tensor = 1.0, u_rest: torch.Tensor = 0.0) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            u (torch.Tensor): 输入电位
            u_threshold (torch.Tensor): 阈电位
            u_rest (torch.Tensor): 静息电位
        Returns:
            o (torch.Tensor): 输出张量
        """
        return _SF.heaviside_rectangular(u - u_threshold, self.a)


class Polynomial(Firing):
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
        return "a=%g" % self.a
    

    def forward(self, u: torch.Tensor, u_threshold: torch.Tensor = 1.0, u_rest: torch.Tensor = 0.0) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            u (torch.Tensor): 输入电位
            u_threshold (torch.Tensor): 阈电位
            u_rest (torch.Tensor): 静息电位
        Returns:
            o (torch.Tensor): 输出张量
        """
        return _SF.heaviside_polynomial(u - u_threshold, self.a)


class Sigmoid(Firing):
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
        return "a=%g" % self.a
    

    def forward(self, u: torch.Tensor, u_threshold: torch.Tensor = 1.0, u_rest: torch.Tensor = 0.0) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            u (torch.Tensor): 输入电位
            u_threshold (torch.Tensor): 阈电位
            u_rest (torch.Tensor): 静息电位
        Returns:
            o (torch.Tensor): 输出张量
        """
        return _SF.heaviside_sigmoid(u - u_threshold, self.a)


class Gaussian(Firing):
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
        return "a=%g" % self.a
    

    def forward(self, u: torch.Tensor, u_threshold: torch.Tensor = 1.0, u_rest: torch.Tensor = 0.0) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            u (torch.Tensor): 输入电位
            u_threshold (torch.Tensor): 阈电位
            u_rest (torch.Tensor): 静息电位
        Returns:
            o (torch.Tensor): 输出张量
        """
        return _SF.heaviside_gaussian(u - u_threshold, self.a)


class Floor(Firing):
    def __init__(self):
        """
        多值脉冲函数，脉冲为电位向下取整。
        """
        super().__init__(
            multi_spikes = True
        )


    def forward(self, u: torch.Tensor, u_threshold: torch.Tensor = 1.0, u_rest: torch.Tensor = 0.0) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            u (torch.Tensor): 输入电位
            u_threshold (torch.Tensor): 阈电位
            u_rest (torch.Tensor): 静息电位
        Returns:
            o (torch.Tensor): 输出张量
        """
        return _SF._multi_firing_floor.apply((u - u_rest) / (u_threshold - u_rest))


class Ceil(Firing):
    def __init__(self):
        """
        多值脉冲函数，脉冲为电位向上取整。
        """
        super().__init__(
            multi_spikes = True
        )


    def forward(self, u: torch.Tensor, u_threshold: torch.Tensor = 1.0, u_rest: torch.Tensor = 0.0) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            u (torch.Tensor): 输入电位
            u_threshold (torch.Tensor): 阈电位
            u_rest (torch.Tensor): 静息电位
        Returns:
            o (torch.Tensor): 输出张量
        """
        return _SF._multi_firing_ceil.apply((u - u_rest) / (u_threshold - u_rest))


class Round(Firing):
    def __init__(self):
        """
        多值脉冲函数，脉冲为电位四舍五入。
        """
        super().__init__(
            multi_spikes = True
        )


    def forward(self, u: torch.Tensor, u_threshold: torch.Tensor = 1.0, u_rest: torch.Tensor = 0.0) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            u (torch.Tensor): 输入电位
            u_threshold (torch.Tensor): 阈电位
            u_rest (torch.Tensor): 静息电位
        Returns:
            o (torch.Tensor): 输出张量
        """
        return _SF._multi_firing_round.apply((u - u_rest) / (u_threshold - u_rest))