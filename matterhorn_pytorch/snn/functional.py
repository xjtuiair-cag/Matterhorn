# -*- coding: UTF-8 -*-
"""
脉冲神经网络的相关函数。
包括Heaviside阶跃函数及其替代梯度等。
"""


import torch
import torch.nn as nn
from typing import List as _List, Tuple as _Tuple, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union, Any as _Any


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


@torch.jit.script
def transpose(x: torch.Tensor, dims: _Optional[_List[int]] = None) -> torch.Tensor:
    """
    转置一个张量。
    Args:
        x (torch.Tensor): 转置前的张量
        dims (int*): 要转置的维度
    Returns:
        x (torch.Tensor): 转置后的张量
    """
    permute_dims: _List[int] = []
    if dims is None:
        for d in range(x.ndim - 1, -1, -1):
            permute_dims.append(d)
    else:
        leading_dims: _List[int] = []
        for d in range(len(dims) - 1, -1, -1):
            leading_dims.append(dims[d])
        trailing_dims: _List[int] = []
        for d in range(x.ndim):
            if d not in dims:
                trailing_dims.append(d)
        permute_dims = leading_dims + trailing_dims
    y: torch.Tensor = x.permute(permute_dims)
    return y


def merge_time_steps_batch_size(tensor: torch.Tensor) -> _Union[torch.Tensor, _Tuple[int, int]]:
    time_steps, batch_size = tensor.shape[:2]
    tensor = tensor.flatten(0, 1)
    return tensor, (time_steps, batch_size)


def split_time_steps_batch_size(tensor: torch.Tensor, time_steps_batch_size: _Tuple[int, int]) -> torch.Tensor:
    time_steps, batch_size = time_steps_batch_size
    assert time_steps * batch_size == tensor.shape[0], "Incorrect shape for splitting."
    tensor = tensor.reshape([time_steps, batch_size] + list(tensor.shape[1:]))
    return tensor


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


@torch.jit.script
def encode_poisson(x: torch.Tensor, precision: float = 1e-5, count: bool = False) -> torch.Tensor:
    """
    泊松编码（速率编码），将值转化为脉冲发放率。
    Args:
        x (torch.Tensor): 数值
        precision (float): 精度，值越小输出的脉冲计数分辨率越高
        count (bool): 是否发送脉冲计数
    Returns:
        y (torch.Tensor): 脉冲序列
    """
    p = torch.clamp(x, 0.0, 1.0 - precision)
    if p.device.type == "mps":
        r = torch.poisson(-torch.log(1.0 - p.cpu())).to(p)
    else:
        r = torch.poisson(-torch.log(1.0 - p))
    if count:
        y = r
    else:
        y = gt(r, torch.zeros_like(r))
    return y


@torch.jit.script
def encode_temporal(x: torch.Tensor, time_steps: int, t_offset: int = 0, prob: float = 1.0) -> torch.Tensor:
    """
    时间编码，将值转化为脉冲发放时间。
    Args:
        x (torch.Tensor): 数值
        time_steps (int): 时间编码的时间步
        t_offset (int): 时间步偏移量，从第几个时间步开始
        prob (float): 是否发送脉冲计数
    Returns:
        y (torch.Tensor): 脉冲序列
    """
    y_seq = []
    for t in range(time_steps):
        f = le(x, torch.full_like(x, t + t_offset))
        r = le(torch.rand_like(x), torch.full_like(x, prob))
        y = f * r
        y_seq.append(y)
    y = torch.stack(y_seq)
    return y


@torch.jit.script
def decode_sum_spike(x: torch.Tensor) -> torch.Tensor:
    """
    总脉冲计数解码，将脉冲转化为脉冲计数。
    Args:
        x (torch.Tensor): 脉冲序列
    Returns:
        y (torch.Tensor): 解码后的脉冲序列
    """
    y = x.sum(dim = 0)
    return y


@torch.jit.script
def decode_avg_spike(x: torch.Tensor) -> torch.Tensor:
    """
    平均脉冲计数解码，将脉冲转化为平均脉冲计数。
    Args:
        x (torch.Tensor): 脉冲序列
    Returns:
        y (torch.Tensor): 解码后的脉冲序列
    """
    y = x.mean(dim = 0)
    return y


@torch.jit.script
def decode_min_time(x: torch.Tensor, t_offset: int, empty_fill: float = -1) -> torch.Tensor:
    """
    最短时间解码，将脉冲转化为首个脉冲的时间。
    Args:
        x (torch.Tensor): 脉冲序列
        t_offset (int): 时间步偏移量，从第几个时间步开始
        empty_fill (float): 如果脉冲序列为全0序列，值应该用什么替代，在TNN中该参数应设为torch.inf
    Returns:
        y (torch.Tensor): 解码后的脉冲序列
    """
    t = (torch.argmax(x, dim = 0) + t_offset).to(x)
    x_sum = x.sum(dim = 0)
    mask = x_sum > 0
    y = torch.where(mask, t, torch.full_like(t, empty_fill))
    return y


@torch.jit.script
def decode_avg_time(x: torch.Tensor, t_offset: int, empty_fill: float = -1) -> torch.Tensor:
    """
    平均脉冲时间解码，将脉冲转化为平均脉冲时间。
    Args:
        x (torch.Tensor): 脉冲序列
        t_offset (int): 时间步偏移量，从第几个时间步开始
        empty_fill (float): 如果脉冲序列为全0序列，值应该用什么替代，在TNN中该参数应设为torch.inf
    Returns:
        y (torch.Tensor): 解码后的脉冲序列
    """
    t = transpose(transpose(x) * torch.arange(x.shape[0]).to(x))
    t_sum = t.sum(dim = 0)
    x_sum = x.sum(dim = 0)
    mask = x_sum > 0
    y = torch.where(mask, t_sum / x_sum + t_offset, torch.full_like(t_sum, empty_fill))
    return y