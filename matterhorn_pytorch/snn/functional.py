# -*- coding: UTF-8 -*-
"""
脉冲神经网络的相关函数。
包括Heaviside阶跃函数及其替代梯度等。
"""


import torch
import torch.nn.functional as _F
from typing import List as _List, Tuple as _Tuple, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union, Any as _Any


class _from_spike_train(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, o: torch.Tensor) -> torch.Tensor:
        """
        脉冲转模拟值的前向传播函数。
        Args:
            x (torch.Tensor): 模拟值
        Returns:
            o (torch.Tensor): 脉冲值（0、1）
        """
        return o


    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        脉冲转模拟值的反向传播函数。
        Args:
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        return grad_output


def from_spike_train(o: torch.Tensor) -> torch.Tensor:
    """
    脉冲转模拟值。
    Args:
        o (torch.Tensor): 脉冲值（0、1）
    Returns:
        x (torch.Tensor): 模拟值
    """
    return _from_spike_train.apply(o)


class _to_spike_train(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor) -> torch.Tensor:
        """
        模拟值转脉冲的前向传播函数。
        Args:
            x (torch.Tensor): 模拟值
        Returns:
            o (torch.Tensor): 脉冲值（0、1）
        """
        return x.gt(0.5).to(x)


    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        模拟值转脉冲的反向传播函数。
        Args:
            grad_output (torch.Tensor): 输出梯度
        Returns:
            grad_input (torch.Tensor): 输入梯度
        """
        return grad_output


def to_spike_train(x: torch.Tensor) -> torch.Tensor:
    """
    模拟值转脉冲。
    Args:
        x (torch.Tensor): 模拟值
    Returns:
        o (torch.Tensor): 脉冲值（0、1）
    """
    return _to_spike_train.apply(x)


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
    h = ((a ** 0.5) / 2.0 - a / 4.0 * torch.abs(x)) * torch.sign(2.0 / (a ** 0.5) - torch.abs(x)) * (torch.abs(x) < (2.0 / (a ** 0.5))).to(x)
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
        ctx.save_for_backward(x)
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
        x, = ctx.saved_tensors
        # Seen as y = torch.log(torch.exp(x - 1.0) + 1.0)
        return grad_output * torch.nn.functional.sigmoid(x - 1.0)


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
        ctx.save_for_backward(x)
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
        x, = ctx.saved_tensors
        # Seen as y = torch.log(torch.exp(x) + 1.0)
        return grad_output * torch.nn.functional.sigmoid(x)


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
        ctx.save_for_backward(x)
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
        x, = ctx.saved_tensors
        # Seen as y = torch.log(torch.exp(x - 0.5) + 1.0)
        return grad_output * torch.nn.functional.sigmoid(x - 0.5)


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
        t_offset (int): 时间步偏移量
        prob (float): 是否发送脉冲计数
    Returns:
        y (torch.Tensor): 脉冲序列
    """
    y = torch.stack([le(x, torch.full_like(x, t + t_offset)) * le(torch.rand_like(x), torch.full_like(x, prob)) for t in range(time_steps)])
    return y


@torch.jit.script
def encode_binary(x: torch.Tensor, length: int = 8, repeat: int = 1) -> torch.Tensor:
    """
    二进制（相位）编码，将值转化为二进制位。
    Args:
        x (torch.Tensor): 数值
        length (int): 二进制位长度
        repeat (int): 重复次数
    Returns:
        y (torch.Tensor): 脉冲序列
    """
    b = torch.where(x >= 0, x, x + (2 << length)).to(torch.long)
    y = torch.stack([(b >> (length - 1 - i)) & 0x1 for i in range(length)] * repeat)
    return y.to(x)


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
        t_offset (int): 时间步偏移量
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
        t_offset (int): 时间步偏移量
        empty_fill (float): 如果脉冲序列为全0序列，值应该用什么替代，在TNN中该参数应设为torch.inf
    Returns:
        y (torch.Tensor): 解码后的脉冲序列
    """
    if x.ndim > 1:
        t = (x.swapaxes(0, -1) * torch.arange(x.shape[0]).to(x)).swapaxes(0, -1)
    else:
        t = x * torch.arange(x.shape[0]).to(x)
    t_sum = t.sum(dim = 0)
    x_sum = x.sum(dim = 0)
    mask = x_sum > 0
    y = torch.where(mask, t_sum / x_sum + t_offset, torch.full_like(t_sum, empty_fill))
    return y


@torch.jit.script
def reset_hard(u: torch.Tensor, o: torch.Tensor, u_rest: torch.Tensor = torch.tensor(0.0)) -> torch.Tensor:
    """
    硬重置（归零重置）。
    Args:
        u (torch.Tensor): 当前电位
        o (torch.Tensor): 当前脉冲
        u_rest (torch.Tensor): 静息电位
    Returns:
        h (torch.Tensor): 当前残余电位
    """
    s = to_spike_train(o)
    h = u * (1.0 - s) + u_rest.to(u) * s
    return h


@torch.jit.script
def reset_soft(u: torch.Tensor, o: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor = torch.tensor(0.0)) -> torch.Tensor:
    """
    软重置（减法重置）。
    Args:
        u (torch.Tensor): 当前电位
        o (torch.Tensor): 当前脉冲
        u_threshold (torch.Tensor): 阈电位
        u_rest (torch.Tensor): 静息电位
    Returns:
        h (torch.Tensor): 当前残余电位
    """
    h = u - o * (u_threshold.to(u) - u_rest.to(u))
    return h


def _firing(u: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, firing: str = "heaviside") -> torch.Tensor:
    """
    脉冲函数。
    Args:
        u (torch.Tensor): 当前电位
        u_threshold (torch.Tensor): 阈电位
        u_rest (torch.Tensor): 静息电位
        firing (str): 选择替代梯度
    Returns:
        o (torch.Tensor): 当前脉冲
    """
    u_threshold = u_threshold.to(u)
    u_rest = u_rest.to(u)
    if firing == "floor":
        return floor((u - u_rest) / (u_threshold - u_rest))
    elif firing == "ceil":
        return ceil((u - u_rest) / (u_threshold - u_rest))
    elif firing == "round":
        return round((u - u_rest) / (u_threshold - u_rest))
    elif firing == "rectangular":
        return heaviside_rectangular(u - u_threshold)
    elif firing == "polynomial":
        return heaviside_polynomial(u - u_threshold)
    elif firing == "sigmoid":
        return heaviside_sigmoid(u - u_threshold)
    else:
        return heaviside_gaussian(u - u_threshold)


@torch.jit.script
def if_neuron(x: torch.Tensor, h: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, firing: str = "heaviside", hard_reset: bool = True) -> _Tuple[torch.Tensor, torch.Tensor]:
    """
    IF神经元。
    Args:
        x (torch.Tensor): 当前输入，形状为[T, B, ...]
        h (torch.Tensor): 初始残余电位，形状为[B, ...]
        u_threshold (torch.Tensor): 阈电位
        u_rest (torch.Tensor): 静息电位
        firing (str): 选择替代梯度
        hard_reset (bool): 重置为硬重置/归零重置(True)还是软重置/减法重置(False)
    Returns:
        o (torch.Tensor): 当前脉冲，形状为[T, B, ...]
        h (torch.Tensor): 最终残余电位，形状为[B, ...]
    """
    o_seq = []
    for t in range(x.shape[0]):
        du = x[t]
        u = h + du
        o = _firing(u, u_threshold, u_rest, firing)
        if hard_reset:
            h = reset_hard(u, o, u_rest)
        else:
            h = reset_soft(u, o, u_threshold, u_rest)
        o_seq.append(o)
    o = torch.stack(o_seq)
    return o, h


@torch.jit.script
def lif_neuron(x: torch.Tensor, h: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, tau_m: torch.Tensor, firing: str = "heaviside", hard_reset: bool = True) -> _Tuple[torch.Tensor, torch.Tensor]:
    """
    LIF神经元。
    Args:
        x (torch.Tensor): 当前输入，形状为[T, B, ...]
        h (torch.Tensor): 初始残余电位，形状为[B, ...]
        u_threshold (torch.Tensor): 阈电位
        u_rest (torch.Tensor): 静息电位
        tau_m (torch.Tensor): 神经元时间常数
        firing (str): 选择替代梯度
        hard_reset (bool): 重置为硬重置/归零重置(True)还是软重置/减法重置(False)
    Returns:
        o (torch.Tensor): 当前脉冲，形状为[T, B, ...]
        h (torch.Tensor): 最终残余电位，形状为[B, ...]
    """
    o_seq = []
    for t in range(x.shape[0]):
        du = (1.0 / tau_m.to(x)) * (-(h - u_rest.to(x)) + x[t])
        u = h + du
        o = _firing(u, u_threshold, u_rest, firing)
        if hard_reset:
            h = reset_hard(u, o, u_rest)
        else:
            h = reset_soft(u, o, u_threshold, u_rest)
        o_seq.append(o)
    o = torch.stack(o_seq)
    return o, h


@torch.jit.script
def qif_neuron(x: torch.Tensor, h: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, tau_m: torch.Tensor, u_c: torch.Tensor, a_0: torch.Tensor, firing: str = "heaviside", hard_reset: bool = True) -> _Tuple[torch.Tensor, torch.Tensor]:
    """
    QIF神经元。
    Args:
        x (torch.Tensor): 当前输入，形状为[T, B, ...]
        h (torch.Tensor): 初始残余电位，形状为[B, ...]
        u_threshold (torch.Tensor): 阈电位
        u_rest (torch.Tensor): 静息电位
        tau_m (torch.Tensor): 神经元时间常数
        u_c (torch.Tensor): 参数$u_{c}$
        a_0 (torch.Tensor): 参数$a_{0}$
        firing (str): 选择替代梯度
        hard_reset (bool): 重置为硬重置/归零重置(True)还是软重置/减法重置(False)
    Returns:
        o (torch.Tensor): 当前脉冲，形状为[T, B, ...]
        h (torch.Tensor): 最终残余电位，形状为[B, ...]
    """
    o_seq = []
    for t in range(x.shape[0]):
        du = (1.0 / tau_m.to(x)) * (a_0.to(x) * (h - u_rest.to(x)) * (h - u_c.to(x)) + x[t])
        u = h + du
        o = _firing(u, u_threshold, u_rest, firing)
        if hard_reset:
            h = reset_hard(u, o, u_rest)
        else:
            h = reset_soft(u, o, u_threshold, u_rest)
        o_seq.append(o)
    o = torch.stack(o_seq)
    return o, h


@torch.jit.script
def expif_neuron(x: torch.Tensor, h: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, tau_m: torch.Tensor, u_t: torch.Tensor, delta_t: torch.Tensor, firing: str = "heaviside", hard_reset: bool = True) -> _Tuple[torch.Tensor, torch.Tensor]:
    """
    ExpIF神经元。
    Args:
        x (torch.Tensor): 当前输入，形状为[T, B, ...]
        h (torch.Tensor): 初始残余电位，形状为[B, ...]
        u_threshold (torch.Tensor): 阈电位
        u_rest (torch.Tensor): 静息电位
        tau_m (torch.Tensor): 神经元时间常数
        u_t (torch.Tensor): 参数$u_{T}$
        delta_t (torch.Tensor): 参数$\Delta_{T}$
        firing (str): 选择替代梯度
        hard_reset (bool): 重置为硬重置/归零重置(True)还是软重置/减法重置(False)
    Returns:
        o (torch.Tensor): 当前脉冲，形状为[T, B, ...]
        h (torch.Tensor): 最终残余电位，形状为[B, ...]
    """
    o_seq = []
    for t in range(x.shape[0]):
        du = (1.0 / tau_m) * (-(h - u_rest) + delta_t * torch.exp((h - u_t) / delta_t) + x)
        u = h + du
        o = _firing(u, u_threshold, u_rest, firing)
        if hard_reset:
            h = reset_hard(u, o, u_rest)
        else:
            h = reset_soft(u, o, u_threshold, u_rest)
        o_seq.append(o)
    o = torch.stack(o_seq)
    return o, h


@torch.jit.script
def izhikevich_neuron(x: torch.Tensor, h: torch.Tensor, w: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, a: torch.Tensor, b: torch.Tensor, firing: str = "heaviside", hard_reset: bool = True) -> _Tuple[torch.Tensor, _Tuple[torch.Tensor, torch.Tensor]]:
    """
    Izhikevich神经元。
    Args:
        x (torch.Tensor): 当前输入，形状为[T, B, ...]
        h (torch.Tensor): 初始残余电位，形状为[B, ...]
        w (torch.Tensor): 初始状态w，形状为[B, ...]
        u_threshold (torch.Tensor): 阈电位
        u_rest (torch.Tensor): 静息电位
        a (torch.Tensor): 参数$a$
        b (torch.Tensor): 参数$b$
        firing (str): 选择替代梯度
        hard_reset (bool): 重置为硬重置/归零重置(True)还是软重置/减法重置(False)
    Returns:
        o (torch.Tensor): 当前脉冲，形状为[T, B, ...]
        h_w (torch.Tensor): 最终残余电位与状态w，形状均为[B, ...]
    """
    o_seq = []
    for t in range(x.shape[0]):
        dw = a * (b * h - w)
        w = w + dw
        du = 0.00004 * h * h + 0.005 * h + 0.14 + u_rest - w + x
        u = h + du
        o = _firing(u, u_threshold, u_rest, firing)
        if hard_reset:
            h = reset_hard(u, o, u_rest)
        else:
            h = reset_soft(u, o, u_threshold, u_rest)
        o_seq.append(o)
    o = torch.stack(o_seq)
    return o, (h, w)


@torch.jit.script
def klif_neuron(x: torch.Tensor, h: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, tau_m: torch.Tensor, k: torch.Tensor, firing: str = "heaviside", hard_reset: bool = True) -> _Tuple[torch.Tensor, torch.Tensor]:
    """
    KLIF神经元。
    Args:
        x (torch.Tensor): 当前输入，形状为[T, B, ...]
        h (torch.Tensor): 初始残余电位，形状为[B, ...]
        u_threshold (torch.Tensor): 阈电位
        u_rest (torch.Tensor): 静息电位
        tau_m (torch.Tensor): 神经元时间常数
        k (torch.Tensor): 常数$k$
        firing (str): 选择替代梯度
        hard_reset (bool): 重置为硬重置/归零重置(True)还是软重置/减法重置(False)
    Returns:
        o (torch.Tensor): 当前脉冲，形状为[T, B, ...]
        h (torch.Tensor): 最终残余电位，形状为[B, ...]
    """
    o_seq = []
    for t in range(x.shape[0]):
        du = (1.0 / tau_m.to(x)) * (-(h - u_rest.to(x)) + x[t])
        u = h + du
        u = _F.relu(k * (u - u_rest)) + u_rest
        o = _firing(u, u_threshold, u_rest, firing)
        if hard_reset:
            h = reset_hard(u, o, u_rest)
        else:
            h = reset_soft(u, o, u_threshold, u_rest)
        o_seq.append(o)
    o = torch.stack(o_seq)
    return o, h


@torch.jit.script
def lim_neuron(x: torch.Tensor, h: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, tau_m: torch.Tensor, firing: str = "heaviside", hard_reset: bool = True) -> _Tuple[torch.Tensor, torch.Tensor]:
    """
    LIM神经元。
    Args:
        x (torch.Tensor): 当前输入，形状为[T, B, ...]
        h (torch.Tensor): 初始残余电位，形状为[B, ...]
        u_threshold (torch.Tensor): 阈电位
        u_rest (torch.Tensor): 静息电位
        tau_m (torch.Tensor): 神经元时间常数
        firing (str): 选择替代梯度
        hard_reset (bool): 重置为硬重置/归零重置(True)还是软重置/减法重置(False)
    Returns:
        u (torch.Tensor): 当前电位，形状为[T, B, ...]
        h (torch.Tensor): 最终残余电位，形状为[B, ...]
    """
    u_seq = []
    for t in range(x.shape[0]):
        du = (1.0 / tau_m.to(x)) * (-(h - u_rest.to(x)) + x[t])
        u = h + du
        o = _firing(u, u_threshold, u_rest, firing)
        if hard_reset:
            h = reset_hard(u, o, u_rest)
        else:
            h = reset_soft(u, o, u_threshold, u_rest)
        u_seq.append(u)
    u = torch.stack(u_seq)
    return u, h