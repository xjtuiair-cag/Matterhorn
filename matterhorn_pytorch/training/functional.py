# -*- coding: UTF-8 -*-
"""
自定义SNN的学习机制，以STDP作为样例。
"""


import torch
import torch.nn as nn
import torch.nn.functional as _F
from typing import Any as _Any, Tuple as _Tuple, Callable as _Callable, Optional as _Optional


@torch.jit.script
def stdp_online(delta_weight: torch.Tensor, input_trace: torch.Tensor, output_trace: torch.Tensor, input_spike_train: torch.Tensor, output_spike_train: torch.Tensor, a_pos: float, tau_pos: float, a_neg: float, tau_neg: float) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在线脉冲时序依赖可塑性（STDP），基于所记录的迹给出更新量。
    Args:
        delta_weight (torch.Tensor): 权重矩阵，形状为[O, I]
        input_trace (torch.Tensor): 输入的迹，形状为[B, O, I]
        output_trace (torch.Tensor): 输出的迹，形状为[B, O, I]
        input_spike_train (torch.Tensor): 当前时间步的输入脉冲，形状为[B, I]
        output_spike_train (torch.Tensor): 当前时间步的输入脉冲，形状为[B, O]
        a_pos (float): STDP参数A+
        tau_pos (float): STDP参数tau+
        a_neg (float): STDP参数A-
        tau_neg (float): STDP参数tau-
    Returns:
        delta_weight (torch.Tensor): 更新权重之后的权重矩阵，形状为[O, I]
        input_trace (torch.Tensor): 输入的迹，形状为[B, O, I]
        output_trace (torch.Tensor): 输出的迹，形状为[B, O, I]
    """
    input_trace_batch_size, input_trace_output_shape, input_trace_input_shape = input_trace.shape
    output_trace_batch_size, output_trace_output_shape, output_trace_input_shape = output_trace.shape
    input_batch_size, input_shape = input_spike_train.shape
    output_batch_size, output_shape = output_spike_train.shape
    weight_output_shape, weight_input_shape = delta_weight.shape

    assert input_shape == weight_input_shape, "Unmatched input shape, %d required for weight but %d found." % (weight_input_shape, input_shape)
    assert input_shape == input_trace_input_shape, "Unmatched input shape, %d required for input trace but %d found." % (input_trace_input_shape, input_shape)
    assert input_shape == output_trace_input_shape, "Unmatched input shape, %d required for output trace but %d found." % (output_trace_input_shape, input_shape)
    assert output_shape == weight_output_shape, "Unmatched output shape, %d required but %d found." % (weight_output_shape, output_shape)
    assert output_shape == input_trace_output_shape, "Unmatched output shape, %d required for input trace but %d found." % (input_trace_output_shape, output_shape)
    assert output_shape == output_trace_output_shape, "Unmatched output shape, %d required for output trace but %d found." % (output_trace_output_shape, output_shape)
    assert output_batch_size == input_batch_size, "Unmatched batch size, %d for output but %d for input." % (output_batch_size, input_batch_size)
    assert input_trace_batch_size == input_batch_size, "Unmatched batch size, %d required for input trace but %d found." % (input_trace_batch_size, input_batch_size)
    assert output_trace_batch_size == output_batch_size, "Unmatched batch size, %d required for output trace but %d found." % (output_trace_batch_size, output_batch_size)
    batch_size = output_batch_size

    input_spike_mat = input_spike_train[:, None].repeat_interleave(output_shape, dim = 1)
    output_spike_mat = output_spike_train[:, :, None].repeat_interleave(input_shape, dim = 2)
    delta_weight += torch.sum(a_pos * output_spike_mat * input_trace - a_neg * input_spike_mat * output_trace, dim = 0)
    input_trace = 1.0 / tau_neg * input_trace + input_spike_mat
    output_trace = 1.0 / tau_pos * output_trace + output_spike_mat
    return delta_weight, input_trace, output_trace


@torch.jit.script
def _f_stdp_linear(input: torch.Tensor, output: torch.Tensor, input_trace: torch.Tensor, output_trace: torch.Tensor, weight: torch.Tensor, a_pos: float, tau_pos: float, a_neg: float, tau_neg: float) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_spike_train = input.clone()
    output_spike_train = output.clone()
    delta_weight = torch.zeros_like(weight)
    T, B, C = input_spike_train.shape
    T, B, N = output_spike_train.shape
    N, C = weight.shape
    for t in range(T):
        delta_weight, input_trace, output_trace = stdp_online(
            delta_weight = delta_weight, # [O, I]
            input_trace = input_trace, # [B, O, I]
            output_trace = output_trace, # [B, O, I]
            input_spike_train = input_spike_train[t], # [B, I]
            output_spike_train = output_spike_train[t], # [B, O]
            a_pos = a_pos,
            tau_pos = tau_pos,
            a_neg = a_neg,
            tau_neg = tau_neg
        )
    return delta_weight, input_trace, output_trace


class _stdp_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, input: torch.Tensor, hidden: _Optional[torch.Tensor], weight: torch.Tensor, input_trace: torch.Tensor, output_trace: torch.Tensor, soma: _Callable, a_pos: float = 0.015, tau_pos: float = 2.0, a_neg: float = 0.015, tau_neg: float = 2.0, training: bool = True) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        利用STDP进行学习的全连接层的前向传播函数。
        Args:
            ctx (Any): 上下文
            input (torch.Tensor): 输入脉冲序列
            hidden (torch.Tensor): 胞体的历史电位
            weight (torch.Tensor): 权重矩阵
            input_trace (torch.Tensor): 输入的迹，累积的输入效应
            output_trace (torch.Tensor): 输出的迹，累积的输出效应
            soma (snn.Module): 胞体
            a_pos (float): STDP参数A+
            tau_pos (float): STDP参数tau+
            a_neg (float): STDP参数A-
            tau_neg (float): STDP参数tau-
            training (bool): 是否正在训练
        Returns:
            output (torch.Tensor): 输出脉冲序列
            hidden (torch.Tensor): 胞体的历史电位
            input_trace (torch.Tensor): 输入的迹，累积的输入效应
            output_trace (torch.Tensor): 输出的迹，累积的输出效应
        """
        ctx.grad_hidden = hidden is not None
        T, B = input.shape[:2]
        psp = _F.linear(input.flatten(0, 1), weight, bias = None)
        psp = psp.unflatten(0, (T, B))
        output, hidden = soma(psp, hidden)
        if training:
            delta_weight, input_trace, output_trace = _f_stdp_linear(input, output, weight, input_trace, output_trace, a_pos, tau_pos, a_neg, tau_neg)
        else:
            delta_weight = torch.zeros_like(weight)
        ctx.save_for_backward(delta_weight, input)
        return output, hidden, input_trace, output_trace


    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor, grad_hidden: torch.Tensor, grad_input_trace: torch.Tensor, grad_output_trace: torch.Tensor) -> _Tuple[torch.Tensor, _Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None]:
        """
        利用STDP进行学习的全连接层的反向传播函数。
        Args:
            ctx (Any): 上下文
            grad_output (torch.Tensor): 输出脉冲序列梯度
            grad_hidden (torch.Tensor): 历史电位的梯度
            grad_input_trace (torch.Tensor): 输入的迹梯度
            grad_output_trace (torch.Tensor): 输出的迹梯度
        Returns:
            grad_input (torch.Tensor): 输入脉冲序列梯度
            grad_hidden (torch.Tensor): 历史电位的梯度
            grad_weight (torch.Tensor): 权重矩阵梯度
            grad_input_trace (torch.Tensor): 输入的迹梯度
            grad_output_trace (torch.Tensor): 输出的迹梯度
            grad_soma (None): 胞体的梯度，为None
            grad_a_pos (None): STDP参数A+的梯度，为None
            grad_tau_pos (None): STDP参数tau+的梯度，为None
            grad_a_neg (None): STDP参数A-的梯度，为None
            grad_tau_neg (None): STDP参数tau-的梯度，为None
            grad_training (None): 是否正在训练的梯度，为None
        """
        delta_weight, input = ctx.saved_tensors
        delta_weight = -delta_weight
        return torch.zeros_like(input), torch.zeros_like(grad_hidden) if ctx.grad_hidden else None, delta_weight, torch.zeros_like(grad_input_trace), torch.zeros_like(grad_output_trace), None, None, None, None, None, None


def stdp_linear(input: torch.Tensor, weight: torch.Tensor, soma: _Callable, hidden: _Optional[torch.Tensor] = None, input_trace: _Optional[torch.Tensor] = None, output_trace: _Optional[torch.Tensor] = None, a_pos: float = 0.015, tau_pos: float = 2.0, a_neg: float = 0.015, tau_neg: float = 2.0, training: bool = True) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    利用STDP进行学习的全连接层。
    Args:
        input (torch.Tensor): 输入脉冲序列
        weight (torch.Tensor): 权重矩阵
        soma (snn.Module): 胞体
        hidden (torch.Tensor | None): 胞体的历史电位
        input_trace (torch.Tensor | None): 输入的迹，累积的输入效应
        output_trace (torch.Tensor | None): 输出的迹，累积的输出效应
        a_pos (float): STDP参数A+
        tau_pos (float): STDP参数tau+
        a_neg (float): STDP参数A-
        tau_neg (float): STDP参数tau-
        training (bool): 是否正在训练
    Returns:
        output (torch.Tensor): 输出脉冲序列
        hidden (torch.Tensor): 胞体的历史电位
        input_trace (torch.Tensor): 输入的迹，累积的输入效应
        output_trace (torch.Tensor): 输出的迹，累积的输出效应
    """
    T, B, C = input.shape
    N, C = weight.shape
    if input_trace is None:
        input_trace = torch.zeros(B, N, C).to(input)
    if output_trace is None:
        output_trace = torch.zeros(B, N, C).to(input)
    return _stdp_linear.apply(input, hidden, weight, input_trace, output_trace, soma, a_pos, tau_pos, a_neg, tau_neg, training)


@torch.jit.script
def _f_stdp_conv2d(input: torch.Tensor, output: torch.Tensor, input_trace: torch.Tensor, output_trace: torch.Tensor, weight: torch.Tensor, stride: _Tuple[int, int], padding: _Tuple[int, int], dilation: _Tuple[int, int], a_pos: float, tau_pos: float, a_neg: float, tau_neg: float) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_spike_train = input.clone()
    output_spike_train = output.clone()
    delta_weight = torch.zeros_like(weight)
    T, B, C, HI, WI = input_spike_train.shape
    T, B, N, HO, WO = output_spike_train.shape
    N, C, HK, WK = weight.shape
    SH, SW = stride
    PH, PW = padding
    DH, DW = dilation
    for y in range(HO):
        for x in range(WO):
            for p in range(HK):
                for q in range(WK):
                    u = y * SH + p * DH - PH
                    v = x * SW + q * DW - PW
                    if u < 0 or u >= HI or v < 0 or v >= WI:
                        continue
                    for t in range(T):
                        delta_weight[:, :, p, q], input_trace[:, :, :, u, v], output_trace[:, :, :, y, x] = stdp_online(
                            delta_weight = delta_weight[:, :, p, q], # [CO, CI]
                            input_trace = input_trace[:, :, :, u, v], # [B, CO, CI]
                            output_trace = output_trace[:, :, :, y, x], # [B, CO, CI]
                            input_spike_train = input_spike_train[t, :, :, u, v], # [B, CI]
                            output_spike_train = output_spike_train[t, :, :, y, x], # [B, CO]
                            a_pos = a_pos,
                            tau_pos = tau_pos,
                            a_neg = a_neg,
                            tau_neg = tau_neg
                        )
    return delta_weight, input_trace, output_trace


class _stdp_conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, input: torch.Tensor, hidden: _Optional[torch.Tensor], weight: torch.Tensor, input_trace: torch.Tensor, output_trace: torch.Tensor, soma: _Callable, stride: _Tuple[int, int], padding: _Tuple[int, int], dilation: _Tuple[int, int], a_pos: float = 0.015, tau_pos: float = 2.0, a_neg: float = 0.015, tau_neg: float = 2.0, training: bool = True) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        利用STDP进行学习的2维卷积层的前向传播函数。
        Args:
            ctx (Any): 上下文
            input (torch.Tensor): 输入脉冲序列
            hidden (torch.Tensor): 胞体的历史电位
            weight (torch.Tensor): 权重矩阵
            input_trace (torch.Tensor): 输入的迹，累积的输入效应
            output_trace (torch.Tensor): 输出的迹，累积的输出效应
            soma (snn.Module): 胞体
            stride (size_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_t): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_t): 卷积的输入步长
            a_pos (float): STDP参数A+
            tau_pos (float): STDP参数tau+
            a_neg (float): STDP参数A-
            tau_neg (float): STDP参数tau-
            training (bool): 是否正在训练
        Returns:
            output (torch.Tensor): 输出脉冲序列
            hidden (torch.Tensor): 胞体的历史电位
            input_trace (torch.Tensor): 输入的迹，累积的输入效应
            output_trace (torch.Tensor): 输出的迹，累积的输出效应
        """
        ctx.grad_hidden = hidden is not None
        T, B = input.shape[:2]
        psp = _F.conv2d(input.flatten(0, 1), weight, bias = None, stride = stride, padding = padding, dilation = dilation)
        psp = psp.unflatten(0, (T, B))
        output, hidden = soma(psp, hidden)
        if training:
            delta_weight, input_trace, output_trace = _f_stdp_conv2d(input, output, input_trace, output_trace, weight, stride, padding, dilation, a_pos, tau_pos, a_neg, tau_neg)
        else:
            delta_weight = torch.zeros_like(weight)
        ctx.save_for_backward(delta_weight, input)
        return output, hidden, input_trace, output_trace


    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor, grad_hidden: torch.Tensor, grad_input_trace: torch.Tensor, grad_output_trace: torch.Tensor) -> _Tuple[torch.Tensor, _Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None, None, None, None]:
        """
        利用STDP进行学习的2维卷积层的反向传播函数。
        Args:
            ctx (Any): 上下文
            grad_output (torch.Tensor): 输出脉冲序列梯度
            grad_hidden (torch.Tensor): 历史电位的梯度
            grad_input_trace (torch.Tensor): 输入的迹梯度
            grad_output_trace (torch.Tensor): 输出的迹梯度
        Returns:
            grad_input (torch.Tensor): 输入脉冲序列梯度
            grad_hidden (torch.Tensor): 历史电位的梯度
            grad_weight (torch.Tensor): 权重矩阵梯度
            grad_input_trace (torch.Tensor): 输入的迹梯度
            grad_output_trace (torch.Tensor): 输出的迹梯度
            grad_soma (None): 胞体的梯度，为None
            grad_stride (size_t): 输出步长的梯度，为None
            grad_padding (size_t): 边缘填充的量的梯度，为None
            grad_dilation (size_t): 输入步长的梯度，为None
            grad_a_pos (None): STDP参数A+的梯度，为None
            grad_tau_pos (None): STDP参数tau+的梯度，为None
            grad_a_neg (None): STDP参数A-的梯度，为None
            grad_tau_neg (None): STDP参数tau-的梯度，为None
            grad_training (None): 是否正在训练的梯度，为None
        """
        delta_weight, input = ctx.saved_tensors
        delta_weight = -delta_weight
        return torch.zeros_like(input), torch.zeros_like(grad_hidden) if ctx.grad_hidden else None, delta_weight, torch.zeros_like(grad_input_trace), torch.zeros_like(grad_output_trace), None, None, None, None, None, None, None, None, None


def stdp_conv2d(input: torch.Tensor, weight: torch.Tensor, soma: _Callable, stride: _Tuple[int, int], padding: _Tuple[int, int], dilation: _Tuple[int, int], hidden: _Optional[torch.Tensor] = None, input_trace: _Optional[torch.Tensor] = None, output_trace: _Optional[torch.Tensor] = None, a_pos: float = 0.015, tau_pos: float = 2.0, a_neg: float = 0.015, tau_neg: float = 2.0, training: bool = True) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    利用STDP进行学习的2维卷积层。
    Args:
        input (torch.Tensor): 输入脉冲序列
        weight (torch.Tensor): 权重矩阵
        soma (snn.Module): 胞体
        stride (size_t): 卷积的输出步长，决定卷积输出的形状
        padding (size_t): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
        dilation (size_t): 卷积的输入步长
        hidden (torch.Tensor | None): 胞体的历史电位
        input_trace (torch.Tensor | None): 输入的迹，累积的输入效应
        output_trace (torch.Tensor | None): 输出的迹，累积的输出效应
        a_pos (float): STDP参数A+
        tau_pos (float): STDP参数tau+
        a_neg (float): STDP参数A-
        tau_neg (float): STDP参数tau-
        training (bool): 是否正在训练
    Returns:
        output (torch.Tensor): 输出脉冲序列
        hidden (torch.Tensor): 胞体的历史电位
        input_trace (torch.Tensor): 输入的迹，累积的输入效应
        output_trace (torch.Tensor): 输出的迹，累积的输出效应
    """
    stride, padding, dilation = tuple(stride), tuple(padding), tuple(dilation)
    T, B, C, HI, WI = input.shape
    N, C, HK, WK = weight.shape
    PH, PW = padding
    SH, SW = stride
    DH, DW = dilation
    HO = (HI + 2 * PH - HK * DH) // SH + 1
    WO = (WI + 2 * PW - WK * DW) // SW + 1
    
    if input_trace is None:
        input_trace = torch.zeros(B, N, C, HI, WI).to(input)
    if output_trace is None:
        output_trace = torch.zeros(B, N, C, HO, WO).to(input)
    return _stdp_conv2d.apply(input, hidden, weight, input_trace, output_trace, soma, stride, padding, dilation, a_pos, tau_pos, a_neg, tau_neg, training)