# -*- coding: UTF-8 -*-
"""
自定义SNN的学习机制，以STDP作为样例。
"""


import torch
import torch.nn as nn
from typing import Tuple as _Tuple


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
def stdp(delta_weight: torch.Tensor, input_spike_train: torch.Tensor, output_spike_train: torch.Tensor, a_pos: float, tau_pos: float, a_neg: float, tau_neg: float) -> torch.Tensor:
    """
    调用在线STDP作为STDP的python版本实现。
    Args:
        delta_weight (torch.Tensor): 权重矩阵，形状为[N, C]
        input_spike_train (torch.Tensor): 输入脉冲序列，形状为[T, B, C]
        output_spike_train (torch.Tensor): 输出脉冲序列，形状为[T, B, N]
        a_pos (float): STDP参数A+
        tau_pos (float): STDP参数tau+
        a_neg (float): STDP参数A-
        tau_neg (float): STDP参数tau-
    Returns:
        delta_weight (torch.Tensor): 权重增量
    """
    input_time_steps, input_batch_size, input_shape = input_spike_train.shape
    output_time_steps, output_batch_size, output_shape = output_spike_train.shape
    weight_output_shape, weight_input_shape = delta_weight.shape

    assert input_shape == weight_input_shape, "Unmatched input shape, %d required but %d found." % (weight_input_shape, input_shape)
    assert output_shape == weight_output_shape, "Unmatched output shape, %d required but %d found." % (weight_output_shape, output_shape)
    assert output_time_steps == input_time_steps, "Unmatched time steps, %d for output but %d for input." % (output_time_steps, input_time_steps)
    assert output_batch_size == input_batch_size, "Unmatched batch size, %d for output but %d for input." % (output_batch_size, input_batch_size)
    time_steps = output_time_steps
    batch_size = output_batch_size

    demo = delta_weight[None].repeat_interleave(batch_size, dim = 0) # [B, O, I]
    input_trace = torch.zeros_like(demo)
    output_trace = torch.zeros_like(demo)
    for t in range(time_steps):
        delta_weight, input_trace, output_trace = stdp_online(
            delta_weight = delta_weight,
            input_trace = input_trace,
            output_trace = output_trace,
            input_spike_train = input_spike_train[t],
            output_spike_train = output_spike_train[t],
            a_pos = a_pos,
            tau_pos = tau_pos,
            a_neg = a_neg,
            tau_neg = tau_neg
        )
    return delta_weight