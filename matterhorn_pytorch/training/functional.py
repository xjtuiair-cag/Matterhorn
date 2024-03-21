# -*- coding: UTF-8 -*-
"""
自定义SNN的学习机制，以STDP作为样例。
"""


import torch
import torch.nn as nn
from typing import Union
try:
    from rich import print
except:
    pass


@torch.jit.script
def stdp_py(delta_weight: torch.Tensor, input_shape: int, output_shape: int, time_steps: int, input_spike_train: torch.Tensor, output_spike_train: torch.Tensor, a_pos: float, tau_pos: float, a_neg: float, tau_neg: float, batch_size: int = 1) -> torch.Tensor:
    """
    STDP的python版本实现，不到万不得已不会调用（性能是灾难级别的）
    Args:
        delta_weight (torch.Tensor): 权重矩阵，形状为[output_shape, input_shape]
        input_shape (int): 输入长度
        output_shape (int): 输出长度
        time_steps (int): 时间步长
        input_spike_train (torch.Tensor): 输入脉冲序列，形状为[input_shape, time_steps]
        output_spike_train (torch.Tensor): 输出脉冲序列，形状为[output_shape, time_steps]
        a_pos (float): STDP参数A+
        tau_pos (float): STDP参数tau+
        a_neg (float): STDP参数A-
        tau_neg (float): STDP参数tau-
    Returns:
        delta_weight (torch.Tensor): 权重增量
    """
    for i in range(output_shape):
        for j in range(input_shape):
            weight = 0.0
            for b in range(batch_size):
                for ti in range(time_steps):
                    if output_spike_train[ti, b, i] < 0.5:
                        continue
                    for tj in range(time_steps):
                        if input_spike_train[tj, b, j] < 0.5:
                            continue
                        dt = ti - tj
                        if dt > 0.0:
                            weight += a_pos * (1.0 / tau_pos) ** (dt - 1.0)
                        elif dt < 0.0:
                            weight -= a_neg * (1.0 / tau_neg) ** (-dt - 1.0)
            delta_weight[i, j] += weight
    return delta_weight


if torch.cuda.is_available():
    try:
        from matterhorn_cuda_extensions import cu_stdp as stdp_cuda
    except:
        stdp_cuda = None
try:
    from matterhorn_cpp_extensions import stdp as stdp_cpp
except:
    stdp_cpp = None


def stdp(delta_weight: torch.Tensor, input_spike_train: torch.Tensor, output_spike_train: torch.Tensor, a_pos: float, tau_pos: float, a_neg: float, tau_neg: float, precision: float = 1e-6) -> torch.Tensor:
    """
    STDP总函数，视情况调用函数
    Args:
        delta_weight (torch.Tensor): 权重矩阵，形状为[output_shape, input_shape]
        input_shape (int): 输入长度
        output_shape (int): 输出长度
        time_steps (int): 时间步长
        input_spike_train (torch.Tensor): 输入脉冲序列，形状为[time_steps, batch_size, input_shape]
        output_spike_train (torch.Tensor): 输出脉冲序列，形状为[time_steps, batch_size, output_shape]
        a_pos (float): STDP参数A+
        tau_pos (float): STDP参数tau+
        a_neg (float): STDP参数A-
        tau_neg (float): STDP参数tau-
        precision (float): 精度，将小于这一数值的值（大多由误差导致）转化为0
    Returns:
        delta_weight: 权重更新值
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

    w_dev = delta_weight.device
    i_dev = input_spike_train.device
    o_dev = output_spike_train.device
    assert w_dev == i_dev, "Unmatched device, %s for weight but %s for input." % (w_dev, i_dev)
    assert w_dev == o_dev, "Unmatched device, %s for weight but %s for output." % (w_dev, o_dev)
    device_type = w_dev.type
    device_idx = w_dev.index

    if device_type == "cuda" and stdp_cuda is not None:
        stdp_cuda(delta_weight, input_shape, output_shape, time_steps, input_spike_train, output_spike_train, a_pos, tau_pos, a_neg, tau_neg, batch_size)
    elif stdp_cpp is not None:
        delta_weight_cpu = delta_weight.cpu()
        stdp_cpp(delta_weight_cpu, input_shape, output_shape, time_steps, input_spike_train.cpu(), output_spike_train.cpu(), a_pos, tau_pos, a_neg, tau_neg, batch_size)
        delta_weight = delta_weight_cpu.to(delta_weight)
    else:
        delta_weight = stdp_py(delta_weight, input_shape, output_shape, time_steps, input_spike_train, output_spike_train, a_pos, tau_pos, a_neg, tau_neg, batch_size)
    delta_weight = torch.where(torch.abs(delta_weight) >= precision, delta_weight, torch.zeros_like(delta_weight))
    return delta_weight