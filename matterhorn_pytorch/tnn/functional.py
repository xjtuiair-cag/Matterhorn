# -*- coding: UTF-8 -*-
"""
TNN的相关函数。
包括时空代数的基本算子等。
"""


import torch
import torch.nn as nn
from typing import Any
from matterhorn_pytorch.snn.functional import forward_heaviside, backward_gaussian, heaviside_gaussian
try:
    from rich import print
except:
    pass


"""
以下是根据时空代数原语完备性定义的算子。
"""
op_min = lambda x, y, min, max, inh: min(x, y)
op_xmin = lambda x, y, min, max, inh: min(op_lt(x, y, min, max, inh), op_lt(y, x, min, max, inh))
op_max = lambda x, y, min, max, inh: max(x, y)
op_xmax = lambda x, y, min, max, inh: min(op_gt(x, y, min, max, inh), op_gt(y, x, min, max, inh))
op_eq = lambda x, y, min, max, inh: max(op_le(x, y, min, max, inh), op_le(y, x, min, max, inh))
op_ne = lambda x, y, min, max, inh: min(op_lt(x, y, min, max, inh), op_gt(x, y, min, max, inh))
op_lt = lambda x, y, min, max, inh: inh(x, y)
op_le = lambda x, y, min, max, inh: inh(x, inh(y, x))
op_gt = lambda x, y, min, max, inh: max(inh(y, x), x)
op_ge = lambda x, y, min, max, inh: inh(x, inh(x, y))


@torch.jit.script
def t_delta(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算TNN中两个脉冲所代表的值（时间）的差值。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = torch.where(torch.isinf(x), torch.full_like(x, torch.inf), torch.where(torch.isinf(y), -torch.full_like(x, torch.inf), x - y))
    return out


def t_to_s(t: torch.Tensor, time_steps: int, t_offset: int = 0) -> torch.Tensor:
    """
    将脉冲时间转换为脉冲序列。
    Args:
        t (torch.Tensor): 时间序列，形状为[...]
        time_steps (int): 最大时间步T
        t_offset (int): 时间步偏移量，从第几个时间步开始，一般为0
    Returns:
        s (torch.Tensor): 脉冲序列，形状为[T, ...]
    """
    t_size = time_steps + t_offset
    T = lambda x: x.permute(*torch.arange(x.ndim - 1, -1, -1))
    spike_ts = torch.ones([t_size] + list(t.shape)) * (t + t_offset)
    current_ts = T(T(torch.ones_like(spike_ts)) * torch.arange(t_size)).to(spike_ts)
    s = heaviside_gaussian(current_ts - spike_ts) # t - ts >= 0 -> t >= ts
    return s


@torch.jit.script
def t_rl_forward_add(x: torch.Tensor, t: int) -> torch.Tensor:
    """
    时间序列竞争逻辑的延迟单元前向传播函数。
    Args:
        x (torch.Tensor): 输入信号
        t (int): 延迟时间
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = x + t
    return out


@torch.jit.script
def t_rl_forward_min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    时间序列竞争逻辑的最小比较单元前向传播函数。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 输入信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = torch.min(x, y)
    return out


@torch.jit.script
def t_rl_forward_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    时间序列竞争逻辑的最大比较单元前向传播函数。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 输入信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = torch.max(x, y)
    return out


@torch.jit.script
def t_rl_forward_inh(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    时间序列竞争逻辑的抑制单元前向传播函数。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    delta = t_delta(x, y)
    mask = delta.lt(0.0)
    out = torch.where(mask, x, torch.full_like(x, torch.inf))
    return out


class t_rl_add(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        时间序列竞争逻辑延迟单元的前向传播函数。
        Args:
            ctx: 上下文
            x (torch.Tensor): 输入信号
            t (int): 延迟时间
        Returns:
            out (torch.Tensor): 输出信号
        """
        out = t_rl_forward_add(x, t)
        return out
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        延迟单元的反向传播（伪）函数。
        Args:
            ctx: 上下文
            grad_output (torch.Tensor): 输出信号梯度
        Returns:
            grad_input (torch.Tensor): 输入信号梯度
        """
        return grad_output, None


class t_rl_min(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        时间序列竞争逻辑最小比较单元的前向传播函数。
        Args:
            ctx: 上下文
            x (torch.Tensor): 输入信号x
            y (torch.Tensor): 输入信号y
        Returns:
            out (torch.Tensor): 输出信号
        """
        out = t_rl_forward_min(x, y)
        ctx.save_for_backward(t_delta(x, out).eq(0.0), t_delta(y, out).eq(0.0))
        return out
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        比较单元的反向传播（伪）函数。
        Args:
            ctx: 上下文
            grad_output (torch.Tensor): 输出信号梯度
        Returns:
            grad_x (torch.Tensor): 输入信号x的梯度
            grad_y (torch.Tensor): 输入信号y的梯度
        """
        x_mask, y_mask = ctx.saved_tensors
        x_mask, y_mask = x_mask.to(grad_output), y_mask.to(grad_output)
        grad_x = grad_output * x_mask
        grad_y = grad_output * y_mask
        return grad_x, grad_y


class t_rl_max(t_rl_min):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        时间序列竞争逻辑最大比较单元的前向传播函数。
        Args:
            ctx: 上下文
            x (torch.Tensor): 输入信号x
            y (torch.Tensor): 输入信号y
        Returns:
            out (torch.Tensor): 输出信号
        """
        out = t_rl_forward_max(x, y)
        ctx.save_for_backward(t_delta(x, out).eq(0.0), t_delta(y, out).eq(0.0))
        return out


class t_rl_inh(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        时间序列竞争逻辑抑制单元的前向传播函数。
        Args:
            ctx: 上下文
            x (torch.Tensor): 输入信号x
            y (torch.Tensor): 抑制信号y
        Returns:
            out (torch.Tensor): 输出信号
        """
        out = t_rl_forward_inh(x, y)
        ctx.save_for_backward(t_delta(x, y))
        return out
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        抑制单元的反向传播（伪）函数。
        Args:
            ctx: 上下文
            grad_output (torch.Tensor): 输出信号梯度
        Returns:
            grad_x (torch.Tensor): 输入信号x的梯度
            grad_y (torch.Tensor): 抑制信号y的梯度
        """
        delta, = ctx.saved_tensors
        mask = ~torch.isinf(delta)
        grad_delta = -backward_gaussian(grad_output, delta)
        grad_x = torch.where(mask, grad_delta, torch.zeros_like(grad_output))
        grad_y = -torch.where(mask, grad_delta, torch.zeros_like(grad_output))
        return grad_x, grad_y


def t_add(x: torch.Tensor, t: int) -> torch.Tensor:
    """
    时间序列的延迟运算符。
    Args:
        x (torch.Tensor): 输入信号
        t (int): 延迟时间
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = t_rl_add.apply(x, t)
    return out


def t_min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    时间序列的最早运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 输入信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_min(x, y, t_rl_min.apply, t_rl_max.apply, t_rl_inh.apply)
    return out


def t_xmin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    时间序列的非同时最早运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 输入信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_xmin(x, y, t_rl_min.apply, t_rl_max.apply, t_rl_inh.apply)
    return out


def t_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    时间序列的最晚运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 输入信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_max(x, y, t_rl_min.apply, t_rl_max.apply, t_rl_inh.apply)
    return out


def t_xmax(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    时间序列的非同时最晚运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 输入信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_xmax(x, y, t_rl_min.apply, t_rl_max.apply, t_rl_inh.apply)
    return out


def t_eq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    时间序列的同时运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_eq(x, y, t_rl_min.apply, t_rl_max.apply, t_rl_inh.apply)
    return out


def t_ne(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    时间序列的非同时运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_ne(x, y, t_rl_min.apply, t_rl_max.apply, t_rl_inh.apply)
    return out


def t_lt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    时间序列的早于运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_lt(x, y, t_rl_min.apply, t_rl_max.apply, t_rl_inh.apply)
    return out


def t_le(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    时间序列的早于或同时运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_le(x, y, t_rl_min.apply, t_rl_max.apply, t_rl_inh.apply)
    return out


def t_gt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    时间序列的晚于运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_gt(x, y, t_rl_min.apply, t_rl_max.apply, t_rl_inh.apply)
    return out


def t_ge(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    时间序列的晚于或同时运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_ge(x, y, t_rl_min.apply, t_rl_max.apply, t_rl_inh.apply)
    return out


def s_to_t(s: torch.Tensor) -> torch.Tensor:
    """
    将脉冲序列转换为脉冲时间。若脉冲序列上无脉冲，则将其转为inf。
    Args:
        s (torch.Tensor): 脉冲序列，形状为[T, ...]
    Returns:
        t (torch.Tensor): 时间序列，形状为[...]
    """
    t = torch.where(torch.sum(s, dim = 0).nonzero(), torch.argmax(s, dim = 0), torch.full_like(s[0], torch.inf))
    return t


@torch.jit.script
def s_rl_forward_add(x: torch.Tensor, t: int) -> torch.Tensor:
    """
    脉冲序列竞争逻辑的延迟单元前向传播函数。
    Args:
        x (torch.Tensor): 输入信号
        t (int): 延迟时间
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = torch.zeros_like(x)
    out[t:] = x[:-t]
    return out


@torch.jit.script
def s_rl_forward_min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    脉冲序列竞争逻辑的最小比较单元前向传播函数。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 输入信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = (x.to(torch.bool) | y.to(torch.bool)).to(x)
    return out


@torch.jit.script
def s_rl_forward_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    脉冲序列竞争逻辑的最大比较单元前向传播函数。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 输入信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = (x.to(torch.bool) & y.to(torch.bool)).to(x)
    return out


@torch.jit.script
def s_rl_forward_inh(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    脉冲序列竞争逻辑的抑制单元前向传播函数。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    time_steps = x.shape[0]
    out_seq = []
    s = torch.zeros_like(x[0]).to(torch.bool)
    for t in range(time_steps):
        i = x[t].to(torch.bool)
        j = y[t].to(torch.bool)
        s = (i & ~j) | s
        o = i & s
        out_seq.append(o)
    out = torch.stack(out_seq).to(x)
    return out


class s_rl_add(t_rl_add):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        脉冲序列竞争逻辑延迟单元的前向传播函数。
        Args:
            ctx: 上下文
            x (torch.Tensor): 输入信号
            t (int): 延迟时间
        Returns:
            out (torch.Tensor): 输出信号
        """
        out = torch.zeros_like(x)
        out[t:] = x[:-t]
        return out


class s_rl_min(t_rl_min):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        脉冲序列竞争逻辑最小比较单元的前向传播函数。
        Args:
            ctx: 上下文
            x (torch.Tensor): 输入信号x
            y (torch.Tensor): 输入信号y
        Returns:
            out (torch.Tensor): 输出信号
        """
        out = s_rl_forward_min(x, y)
        x_t = s_to_t(x)
        y_t = s_to_t(y)
        out_t = s_to_t(out)
        x_mask = torch.stack([t_delta(x_t, out_t).eq(0.0)] * x.shape[0])
        y_mask = torch.stack([t_delta(y_t, out_t).eq(0.0)] * y.shape[0])
        ctx.save_for_backward(x_mask, y_mask)
        return out


class s_rl_max(t_rl_max):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        脉冲序列竞争逻辑最大比较单元的前向传播函数。
        Args:
            ctx: 上下文
            x (torch.Tensor): 输入信号x
            y (torch.Tensor): 输入信号y
        Returns:
            out (torch.Tensor): 输出信号
        """
        out = s_rl_forward_max(x, y)
        x_t = s_to_t(x)
        y_t = s_to_t(y)
        out_t = s_to_t(out)
        x_mask = torch.stack([t_delta(x_t, out_t).eq(0.0)] * x.shape[0])
        y_mask = torch.stack([t_delta(y_t, out_t).eq(0.0)] * y.shape[0])
        ctx.save_for_backward(x_mask, y_mask)
        return out


class s_rl_inh(t_rl_inh):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        脉冲序列竞争逻辑抑制单元的前向传播函数。
        Args:
            ctx: 上下文
            x (torch.Tensor): 输入信号x
            y (torch.Tensor): 抑制信号y
        Returns:
            out (torch.Tensor): 输出信号
        """
        out = s_rl_forward_inh(x, y)
        x_t = s_to_t(x)
        y_t = s_to_t(y)
        delta = torch.stack([t_delta(x_t, y_t)] * x.shape[0])
        ctx.save_for_backward(delta)
        return out


def s_add(x: torch.Tensor, t: int) -> torch.Tensor:
    """
    脉冲序列的延迟运算符。
    Args:
        x (torch.Tensor): 输入信号
        t (int): 延迟时间
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = s_rl_add.apply(x, t)
    return out


def s_min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    脉冲序列的最早运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 输入信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_min(x, y, s_rl_min.apply, s_rl_max.apply, s_rl_inh.apply)
    return out


def s_xmin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    脉冲序列的非同时最早运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 输入信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_xmin(x, y, s_rl_min.apply, s_rl_max.apply, s_rl_inh.apply)
    return out


def s_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    脉冲序列的最晚运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 输入信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_max(x, y, s_rl_min.apply, s_rl_max.apply, s_rl_inh.apply)
    return out


def s_xmax(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    脉冲序列的非同时最晚运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 输入信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_xmax(x, y, s_rl_min.apply, s_rl_max.apply, s_rl_inh.apply)
    return out


def s_eq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    脉冲序列的同时运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_eq(x, y, s_rl_min.apply, s_rl_max.apply, s_rl_inh.apply)
    return out


def s_ne(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    脉冲序列的非同时运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_ne(x, y, s_rl_min.apply, s_rl_max.apply, s_rl_inh.apply)
    return out


def s_lt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    脉冲序列的早于运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_lt(x, y, s_rl_min.apply, s_rl_max.apply, s_rl_inh.apply)
    return out


def s_le(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    脉冲序列的早于或同时运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_le(x, y, s_rl_min.apply, s_rl_max.apply, s_rl_inh.apply)
    return out


def s_gt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    脉冲序列的晚于运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_gt(x, y, s_rl_min.apply, s_rl_max.apply, s_rl_inh.apply)
    return out


def s_ge(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    脉冲序列的晚于或同时运算符。
    Args:
        x (torch.Tensor): 输入信号x
        y (torch.Tensor): 抑制信号y
    Returns:
        out (torch.Tensor): 输出信号
    """
    out = op_ge(x, y, s_rl_min.apply, s_rl_max.apply, s_rl_inh.apply)
    return out