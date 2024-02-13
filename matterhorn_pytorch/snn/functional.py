# -*- coding: UTF-8 -*-
"""
脉冲神经网络的相关函数。
包括Heaviside阶跃函数及其替代梯度等。
"""


import torch
import torch.nn as nn
from typing import Any
try:
    from rich import print
except:
    pass


class _val_to_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        模拟值转脉冲的前向传播函数，以0.5为界
        Args:
            x (torch.Tensor): 模拟值
        Returns:
            o (torch.Tensor): 脉冲值（0、1）
        """
        return x.ge(0.5).to(x)


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        模拟值转脉冲的反向传播函数
        Args:
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        return grad_output


def val_to_spike(x: torch.Tensor) -> torch.Tensor:
    """
    模拟值转脉冲，以0.5为界
    Args:
        x (torch.Tensor): 模拟值
    Returns:
        o (torch.Tensor): 脉冲值（0、1）
    """
    return _val_to_spike.apply(x)


@torch.jit.script
def forward_heaviside(x: torch.Tensor) -> torch.Tensor:
    """
    阶跃函数。当输入大于等于0时，其输出为1；当输入小于0时，其输出为0。
    Args:
        x (torch.Tensor): 输入x
    Returns:
        y (torch.Tensor): 输出u(x)
    """
    return x.ge(0.0).to(x)


@torch.jit.script
def backward_rectangular(grad_output: torch.Tensor, x: torch.Tensor, a: float) -> torch.Tensor:
    """
    阶跃函数的导数，矩形窗，
    详见文章[Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)。
    Args:
        grad_output (torch.Tensor): 输出梯度
        x (torch.Tensor): 输入
        a (float): 参数a，详见文章
    Returns:
        grad_input (torch.Tensor): 输入梯度
    """
    h = (1.0 / a) * torch.logical_and(x.gt(-a / 2.0), x.lt(a / 2.0)).to(x)
    return h * grad_output


class _heaviside_rectangular(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, a: float) -> torch.Tensor:
        """
        使用Heaviside阶跃函数作为前向传播函数。
        Args:
            ctx: 上下文
            x (torch.Tensor): 输入
            a (float): 参数a
        Returns:
            y (torch.Tensor): 输出
        """
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.a = a
        return forward_heaviside(x)
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        使用矩形函数作为反向传播函数。
        Args:
            ctx: 上下文
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        x, = ctx.saved_tensors
        return backward_rectangular(grad_output, x, ctx.a), None


def heaviside_rectangular(x: torch.Tensor, a: float) -> torch.Tensor:
    """
    Heaviside阶跃函数，使用矩形函数作为反向传播函数。
    Args:
        x (torch.Tensor): 模拟值
    Returns:
        o (torch.Tensor): 脉冲值（0、1）
    """
    return _heaviside_rectangular.apply(x, a)


@torch.jit.script
def backward_polynomial(grad_output: torch.Tensor, x: torch.Tensor, a: float) -> torch.Tensor:
    """
    阶跃函数的导数，一次函数窗，
    详见文章[Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)。
    Args:
        grad_output (torch.Tensor): 输出梯度
        x (torch.Tensor): 输入
        a (float): 参数a，详见文章
    Returns:
        grad_input (torch.Tensor): 输入梯度
    """
    h = ((a ** 0.5) / 2.0 - a / 4.0 * torch.abs(x)) * torch.sign(2.0 / (a ** 0.5) - torch.abs(x)) * (torch.abs(x) < (2.0 / (a ** 0.5))).float()
    return h * grad_output


class _heaviside_polynomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, a: float) -> torch.Tensor:
        """
        使用Heaviside阶跃函数作为前向传播函数。
        Args:
            ctx: 上下文
            x (torch.Tensor): 输入
            a (float): 参数a
        Returns:
            y (torch.Tensor): 输出
        """
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.a = a
        return forward_heaviside(x)
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        使用多项式函数作为反向传播函数。
        Args:
            ctx: 上下文
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        x, = ctx.saved_tensors
        return backward_polynomial(grad_output, x, ctx.a), None


def heaviside_polynomial(x: torch.Tensor, a: float) -> torch.Tensor:
    """
    Heaviside阶跃函数，使用多项式函数作为反向传播函数。
    Args:
        x (torch.Tensor): 模拟值
    Returns:
        o (torch.Tensor): 脉冲值（0、1）
    """
    return _heaviside_polynomial.apply(x, a)


@torch.jit.script
def backward_sigmoid(grad_output: torch.Tensor, x: torch.Tensor, a: float) -> torch.Tensor:
    """
    阶跃函数的导数，sigmoid函数窗，
    详见文章[Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)。
    Args:
        grad_output (torch.Tensor): 输出梯度
        x (torch.Tensor): 输入
        a (float): 参数a，详见文章
    Returns:
        grad_input (torch.Tensor): 输入梯度
    """
    ex = torch.exp(-x / a)
    h = (1.0 / a) * (ex / ((1.0 + ex) ** 2.0))
    return h * grad_output


class _heaviside_sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, a: float) -> torch.Tensor:
        """
        使用Heaviside阶跃函数作为前向传播函数。
        Args:
            ctx: 上下文
            x (torch.Tensor): 输入
            a (float): 参数a
        Returns:
            y (torch.Tensor): 输出
        """
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.a = a
        return forward_heaviside(x)
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        使用Sigmoid函数的导数作为反向传播函数。
        Args:
            ctx: 上下文
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        x, = ctx.saved_tensors
        return backward_sigmoid(grad_output, x, ctx.a), None


def heaviside_sigmoid(x: torch.Tensor, a: float) -> torch.Tensor:
    """
    Heaviside阶跃函数，使用Sigmoid函数的导数作为反向传播函数。
    Args:
        x (torch.Tensor): 模拟值
    Returns:
        o (torch.Tensor): 脉冲值（0、1）
    """
    return _heaviside_sigmoid.apply(x, a)


@torch.jit.script
def backward_gaussian(grad_output: torch.Tensor, x: torch.Tensor, a: float) -> torch.Tensor:
    """
    阶跃函数的导数，高斯函数窗，
    详见文章[Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)。
    Args:
        grad_output (torch.Tensor): 输出梯度
        x (torch.Tensor): 输入
        a (float): 参数a，详见文章
    Returns:
        grad_input (torch.Tensor): 输入梯度
    """
    h = (1.0 / ((2.0 * torch.pi * a) ** 0.5)) * torch.exp(-(x ** 2.0) / (2 * a))
    return h * grad_output


class _heaviside_gaussian(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, a: float) -> torch.Tensor:
        """
        使用Heaviside阶跃函数作为前向传播函数。
        Args:
            ctx: 上下文
            x (torch.Tensor): 输入
            a (float): 参数a
        Returns:
            y (torch.Tensor): 输出
        """
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.a = a
        return forward_heaviside(x)
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        使用高斯函数作为反向传播函数。
        Args:
            ctx: 上下文
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        x, = ctx.saved_tensors
        return backward_gaussian(grad_output, x, ctx.a), None


def heaviside_gaussian(x: torch.Tensor, a: float) -> torch.Tensor:
    """
    Heaviside阶跃函数，使用高斯函数作为反向传播函数。
    Args:
        x (torch.Tensor): 模拟值
    Returns:
        o (torch.Tensor): 脉冲值（0、1）
    """
    return _heaviside_gaussian.apply(x, a)