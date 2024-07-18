# -*- coding: UTF-8 -*-
"""
脉冲神经网络的相关函数。
包括Heaviside阶跃函数及其替代梯度等。
"""


import torch
import torch.nn as nn
from typing import Tuple as _Tuple, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union, Any as _Any


def to(u: _Optional[_Union[torch.Tensor, int, float]], x: torch.Tensor) -> torch.Tensor:
    """
    校正张量形状。
    Args:
        u (torch.Tensor): 待校正的数据，可能是张量或浮点值
        x (torch.Tensor): 带有正确数据类型、所在设备和形状的张量
    Returns:
        u (torch.Tensor): 经过校正的张量
    """
    if isinstance(u, torch.Tensor):
        return u.to(x)
    elif isinstance(u, int) or isinstance(u, float):
        return torch.full_like(x, u)
    return torch.zeros_like(x)


def merge_time_steps_batch_size(tensors: _Union[torch.Tensor, _Tuple[torch.Tensor]], tensor_map: _Optional[_Mapping[str, torch.Tensor]] = None) -> _Tuple[_Iterable, _Mapping, _Iterable]:
    if not isinstance(tensors, _Tuple):
        tensors = (tensors,)
    time_steps = tensors[0].shape[0]
    batch_size = tensors[0].shape[1]
    tensors = (tensor.flatten(0, 1) if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors)
    if tensor_map is not None:
        tensor_map = {name: tensor.flatten(0, 1) if isinstance(tensor, torch.Tensor) else tensor for name, tensor in tensor_map.items()}
    return tensors, tensor_map, [time_steps, batch_size]


def split_time_steps_batch_size(tensors: _Union[torch.Tensor, _Tuple[torch.Tensor]], time_steps_batch_size: _Iterable) -> _Union[torch.Tensor, _Tuple[torch.Tensor]]:
    if isinstance(tensors, _Tuple):
        tensors = (tensor.reshape(list(time_steps_batch_size) + list(tensor.shape[1:])) if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors)
    else:
        tensors = tensors.reshape(list(time_steps_batch_size) + list(tensors.shape[1:])) if isinstance(tensors, torch.Tensor) else tensors
    return tensors


class _val_to_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor) -> torch.Tensor:
        """
        模拟值转脉冲的前向传播函数，以0.5为界
        Args:
            x (torch.Tensor): 模拟值
        Returns:
            o (torch.Tensor): 脉冲值（0、1）
        """
        return abs(x).ge(0.5).to(x)


    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor) -> torch.Tensor:
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
def fp_heaviside(x: torch.Tensor) -> torch.Tensor:
    """
    阶跃函数。当输入大于等于0时，其输出为1；当输入小于0时，其输出为0。
    Args:
        x (torch.Tensor): 输入x
    Returns:
        y (torch.Tensor): 输出u(x)
    """
    return x.ge(0.0).to(x)


@torch.jit.script
def bp_rectangular(x: torch.Tensor, a: float) -> torch.Tensor:
    """
    阶跃函数的导数，矩形窗，
    详见文章[Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)。
    Args:
        x (torch.Tensor): 输入
        a (float): 参数a，详见文章
    Returns:
        grad_input (torch.Tensor): 输入梯度
    """
    h = (1.0 / a) * torch.logical_and(x.gt(-a / 2.0), x.lt(a / 2.0)).to(x)
    return h


class _heaviside_rectangular(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor, a: float) -> torch.Tensor:
        """
        使用Heaviside阶跃函数作为前向传播函数。
        Args:
            ctx (Any): 上下文
            x (torch.Tensor): 输入
            a (float): 参数a
        Returns:
            y (torch.Tensor): 输出
        """
        if x.requires_grad:
            ctx.save_for_backward(bp_rectangular(x, a))
        return fp_heaviside(x)
    

    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        使用矩形函数作为反向传播函数。
        Args:
            ctx (Any): 上下文
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        x, = ctx.saved_tensors
        return grad_output * x, None


def heaviside_rectangular(x: torch.Tensor, a: float = 1.0) -> torch.Tensor:
    """
    Heaviside阶跃函数，使用矩形函数作为反向传播函数。
    Args:
        x (torch.Tensor): 模拟值
    Returns:
        o (torch.Tensor): 脉冲值（0、1）
    """
    return _heaviside_rectangular.apply(x, a)


@torch.jit.script
def bp_polynomial(x: torch.Tensor, a: float) -> torch.Tensor:
    """
    阶跃函数的导数，一次函数窗，
    详见文章[Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)。
    Args:
        x (torch.Tensor): 输入
        a (float): 参数a，详见文章
    Returns:
        grad_input (torch.Tensor): 输入梯度
    """
    h = ((a ** 0.5) / 2.0 - a / 4.0 * torch.abs(x)) * torch.sign(2.0 / (a ** 0.5) - torch.abs(x)) * (torch.abs(x) < (2.0 / (a ** 0.5))).float()
    return h


class _heaviside_polynomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor, a: float) -> torch.Tensor:
        """
        使用Heaviside阶跃函数作为前向传播函数。
        Args:
            ctx (Any): 上下文
            x (torch.Tensor): 输入
            a (float): 参数a
        Returns:
            y (torch.Tensor): 输出
        """
        if x.requires_grad:
            ctx.save_for_backward(bp_polynomial(x, a))
        return fp_heaviside(x)
    

    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        使用多项式函数作为反向传播函数。
        Args:
            ctx (Any): 上下文
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        x, = ctx.saved_tensors
        return grad_output * x, None


def heaviside_polynomial(x: torch.Tensor, a: float = 4.0) -> torch.Tensor:
    """
    Heaviside阶跃函数，使用多项式函数作为反向传播函数。
    Args:
        x (torch.Tensor): 模拟值
    Returns:
        o (torch.Tensor): 脉冲值（0、1）
    """
    return _heaviside_polynomial.apply(x, a)


@torch.jit.script
def bp_sigmoid(x: torch.Tensor, a: float) -> torch.Tensor:
    """
    阶跃函数的导数，sigmoid函数窗，
    详见文章[Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)。
    Args:
        x (torch.Tensor): 输入
        a (float): 参数a，详见文章
    Returns:
        grad_input (torch.Tensor): 输入梯度
    """
    ex = torch.exp(-x / a)
    h = (1.0 / a) * (ex / ((1.0 + ex) ** 2.0))
    return h


class _heaviside_sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor, a: float) -> torch.Tensor:
        """
        使用Heaviside阶跃函数作为前向传播函数。
        Args:
            ctx (Any): 上下文
            x (torch.Tensor): 输入
            a (float): 参数a
        Returns:
            y (torch.Tensor): 输出
        """
        if x.requires_grad:
            ctx.save_for_backward(bp_sigmoid(x, a))
        return fp_heaviside(x)
    

    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        使用Sigmoid函数的导数作为反向传播函数。
        Args:
            ctx (Any): 上下文
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        x, = ctx.saved_tensors
        return grad_output * x, None


def heaviside_sigmoid(x: torch.Tensor, a: float = 0.25) -> torch.Tensor:
    """
    Heaviside阶跃函数，使用Sigmoid函数的导数作为反向传播函数。
    Args:
        x (torch.Tensor): 模拟值
    Returns:
        o (torch.Tensor): 脉冲值（0、1）
    """
    return _heaviside_sigmoid.apply(x, a)


@torch.jit.script
def bp_gaussian(x: torch.Tensor, a: float) -> torch.Tensor:
    """
    阶跃函数的导数，高斯函数窗，
    详见文章[Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)。
    Args:
        x (torch.Tensor): 输入
        a (float): 参数a，详见文章
    Returns:
        grad_input (torch.Tensor): 输入梯度
    """
    h = (1.0 / ((2.0 * torch.pi * a) ** 0.5)) * torch.exp(-(x ** 2.0) / (2 * a))
    return h


class _heaviside_gaussian(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor, a: float) -> torch.Tensor:
        """
        使用Heaviside阶跃函数作为前向传播函数。
        Args:
            ctx (Any): 上下文
            x (torch.Tensor): 输入
            a (float): 参数a
        Returns:
            y (torch.Tensor): 输出
        """
        if x.requires_grad:
            ctx.save_for_backward(bp_gaussian(x, a))
        return fp_heaviside(x)
    

    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        使用高斯函数作为反向传播函数。
        Args:
            ctx (Any): 上下文
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        x, = ctx.saved_tensors
        return grad_output * x, None


def heaviside_gaussian(x: torch.Tensor, a: float = 0.16) -> torch.Tensor:
    """
    Heaviside阶跃函数，使用高斯函数作为反向传播函数。
    Args:
        x (torch.Tensor): 模拟值
    Returns:
        o (torch.Tensor): 脉冲值（0、1）
    """
    return _heaviside_gaussian.apply(x, a)


def lt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    可以求导的小于函数。
    Args:
        x (torch.Tensor): 被比较数 x
        y (torch.Tensor): 比较数 y
    Returns:
        res (torch.Tensor): 比较结果
    """
    return 1.0 - ge(x, y)


def le(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    可以求导的小于等于函数。
    Args:
        x (torch.Tensor): 被比较数 x
        y (torch.Tensor): 比较数 y
    Returns:
        res (torch.Tensor): 比较结果
    """
    return heaviside_gaussian(y - x)


def gt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    可以求导的大于函数。
    Args:
        x (torch.Tensor): 被比较数 x
        y (torch.Tensor): 比较数 y
    Returns:
        res (torch.Tensor): 比较结果
    """
    return 1.0 - le(x, y)


def ge(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    可以求导的大于等于函数。
    Args:
        x (torch.Tensor): 被比较数 x
        y (torch.Tensor): 比较数 y
    Returns:
        res (torch.Tensor): 比较结果
    """
    return heaviside_gaussian(x - y)
    

def eq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    可以求导的等于函数。
    Args:
        x (torch.Tensor): 被比较数 x
        y (torch.Tensor): 比较数 y
    Returns:
        res (torch.Tensor): 比较结果
    """
    return le(x, y) * ge(x, y)


def ne(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    可以求导的不等于函数。
    Args:
        x (torch.Tensor): 被比较数 x
        y (torch.Tensor): 比较数 y
    Returns:
        res (torch.Tensor): 比较结果
    """
    return 1.0 - eq(x, y)


class _multi_firing_floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor) -> torch.Tensor:
        """
        多值脉冲函数的前向传播函数。
        Args:
            ctx (Any): 上下文
            x (torch.Tensor): 输入
        Returns:
            y (torch.Tensor): 输出
        """
        ctx.save_for_backward(x.ge(0.0))
        return torch.max(torch.floor(x), torch.zeros_like(x))
    

    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        多值脉冲函数的反向传播函数。
        Args:
            ctx (Any): 上下文
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        mask, = ctx.saved_tensors
        return grad_output * mask.to(grad_output)


class _multi_firing_ceil(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor) -> torch.Tensor:
        """
        多值脉冲函数的前向传播函数。
        Args:
            ctx (Any): 上下文
            x (torch.Tensor): 输入
        Returns:
            y (torch.Tensor): 输出
        """
        ctx.save_for_backward(x.gt(0.0))
        return torch.max(torch.ceil(x), torch.zeros_like(x))
    

    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        多值脉冲函数的反向传播函数。
        Args:
            ctx (Any): 上下文
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        mask, = ctx.saved_tensors
        return grad_output * mask.to(grad_output)


class _multi_firing_round(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor) -> torch.Tensor:
        """
        多值脉冲函数的前向传播函数。
        Args:
            ctx (Any): 上下文
            x (torch.Tensor): 输入
        Returns:
            y (torch.Tensor): 输出
        """
        ctx.save_for_backward(x.gt(0.0))
        return torch.max(torch.round(x), torch.zeros_like(x))
    

    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        多值脉冲函数的反向传播函数。
        Args:
            ctx (Any): 上下文
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        mask, = ctx.saved_tensors
        return grad_output * mask.to(grad_output)


def floor(x: torch.Tensor) -> torch.Tensor:
    """
    可以求导的向下取整函数。
    Args:
        x (torch.Tensor): 被比较数 x
        y (torch.Tensor): 比较数 y
    Returns:
        res (torch.Tensor): 比较结果
    """
    return _multi_firing_floor.apply(x)


def ceil(x: torch.Tensor) -> torch.Tensor:
    """
    可以求导的向上取整函数。
    Args:
        x (torch.Tensor): 被比较数 x
        y (torch.Tensor): 比较数 y
    Returns:
        res (torch.Tensor): 比较结果
    """
    return _multi_firing_ceil.apply(x)


def round(x: torch.Tensor) -> torch.Tensor:
    """
    可以求导的四舍五入函数。
    Args:
        x (torch.Tensor): 被比较数 x
        y (torch.Tensor): 比较数 y
    Returns:
        res (torch.Tensor): 比较结果
    """
    return _multi_firing_round.apply(x)