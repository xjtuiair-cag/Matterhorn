# -*- coding: UTF-8 -*-
"""
液体状态机。
使用邻接矩阵表示神经元的连接方向。
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable
from matterhorn_pytorch import snn
import matterhorn_pytorch.snn.functional as SF
from matterhorn_pytorch.snn.container import Temporal
from matterhorn_pytorch.snn.skeleton import Module
from matterhorn_pytorch.training.functional import stdp_online
try:
    from rich import print
except:
    pass


class LSM(snn.Module):
    def __init__(self, adjacent: torch.Tensor, soma: snn.Module, multi_time_step: bool = True, reset_after_process: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        液体状态机。
        Args:
            adjacent (torch.Tensor): 邻接矩阵，行为各个神经元的轴突，列为各个神经元的树突，1为有从轴突指向树突的连接，0为没有
            soma (nn.Module): 使用的脉冲神经元胞体，在matterhorn_pytorch.snn.soma中选择
            multi_time_step (bool): 是否调整为多个时间步模式
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        self.y_0 = None
        super().__init__(
            multi_time_step = multi_time_step,
            reset_after_process = reset_after_process
        )
        assert len(adjacent.shape) == 2 and adjacent.shape[0] == adjacent.shape[1], "Incorrect adjacent matrix."
        self.adjacent = nn.Parameter(adjacent.to(torch.float), requires_grad = False)
        self.neuron_num = self.adjacent.shape[0]
        self.input_adjacent = nn.Parameter(torch.eye(self.neuron_num, device = device, dtype = dtype), requires_grad = False)
        self.recurrent_weight = nn.Parameter(torch.empty((self.neuron_num, self.neuron_num), device = device, dtype = dtype))
        nn.init.kaiming_uniform_(self.recurrent_weight, a = 5.0 ** 0.5)
        self.input_weight = nn.Parameter(torch.empty((self.neuron_num, self.neuron_num), device = device, dtype = dtype))
        nn.init.kaiming_uniform_(self.input_weight, a = 5.0 ** 0.5)
        if self.multi_time_step:
            if soma.supports_multi_time_step():
                self.soma = soma.multi_time_step_(True)
            elif not soma.multi_time_step:
                self.soma = Temporal(soma, reset_after_process = False)
        else:
            if soma.supports_single_time_step():
                self.soma = soma.multi_time_step_(False)
            else:
                self.soma = soma
        self.reset()


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return "neuron_num=%d, adjacent=\n%s\n, multi_time_step=%s" % (self.neuron_num, self.adjacent, str(self.multi_time_step))


    def reset(self) -> Module:
        """
        重置整个神经元。
        """
        if isinstance(self.soma, snn.Module):
            self.soma.reset()
        self.y_0 = SF.reset_tensor(self.y_0, 0.0)
        return super().reset()


    def f_synapse(self, y_0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_0 = SF.init_tensor(y_0, x)
        y = F.linear(y_0, self.recurrent_weight * self.adjacent.T, None) + F.linear(x, self.input_weight * self.input_adjacent.T, None)
        return y


    def forward_single_time_step(self, x: torch.Tensor, y_0: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 当前输入，形状为[B, I]
            y_0 (torch.Tensor): 上一时刻的输出，形状为[B, O]
        Returns:
            y (torch.Tensor): 当前输出，形状为[B, O]
        """
        x = self.f_synapse(y_0, x)
        y = self.soma(x)
        return y


    def forward_multi_time_step(self, x: torch.Tensor, y_0: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 输入序列，形状为[T, B, I]
            y_0 (torch.Tensor): 初始的输出，形状为[B, O]
        Returns:
            y (torch.Tensor): 输出序列，形状为[T, B, O]
            y_0 (torch.Tensor): 最后的输出，形状为[B, O]
        """
        time_steps = x.shape[0]
        y_seq = []
        for t in range(time_steps):
            y_0 = self.forward_single_time_step(x[t], y_0)
            y_seq.append(y_0)
        y = torch.stack(y_seq)
        return y, y_0


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 当前输入，形状为[B, I]（单步）或[T, B, I]（多步）
        Returns:
            y (torch.Tensor): 当前输出，形状为[B, O]（单步）或[T, B, O]（多步）
        """
        if self.multi_time_step:
            y, self.y_0 = self.forward_multi_time_step(x, self.y_0)
        else:
            y = self.forward_single_time_step(x, self.y_0)
            self.y_0 = y
        return y


class f_stdp_lsm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.Any, input: torch.Tensor, output_0: torch.Tensor, weight: torch.Tensor, input_trace: torch.Tensor, output_trace: torch.Tensor, recurrent_weight: torch.Tensor, recurrent_input_trace: torch.Tensor, recurrent_output_trace: torch.Tensor, forward_func: Callable, a_pos: float = 1.0, tau_pos: float = 2.0, a_neg: float = 1.0, tau_neg: float = 2.0, training: bool = True, multi_time_step: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        利用STDP进行学习的液体状态机的前向传播函数。
        Args:
            ctx (torch.Any): 上下文
            input (torch.Tensor): 输入脉冲序列
            output_0 (torch.Tensor): 初始状态下的循环脉冲
            weight (torch.Tensor): 权重矩阵
            input_trace (torch.Tensor): 输入的迹，累积的输入效应
            output_trace (torch.Tensor): 输出的迹，累积的输出效应
            recurrent_weight (torch.Tensor): 循环权重矩阵
            recurrent_input_trace (torch.Tensor): 在神经元间循环的输入的迹，累积的内输入效应
            recurrent_input_trace (torch.Tensor): 在神经元间循环的输出的迹，累积的内输出效应
            forward_func (Callable): 前向传播函数，由该时刻输入和上一时刻输出得到该时刻输出的函数
            a_pos (float): STDP参数A+
            tau_pos (float): STDP参数tau+
            a_neg (float): STDP参数A-
            tau_neg (float): STDP参数tau-
            training (bool): 是否正在训练
            multi_time_step (bool): 是否为多时间步模式
        Returns:
            output (torch.Tensor): 输出脉冲序列
            output_last (torch.Tensor): 最终的循环脉冲
            input_trace (torch.Tensor): 输入的迹，累积的输入效应
            output_trace (torch.Tensor): 输出的迹，累积的输出效应
            recurrent_input_trace (torch.Tensor): 在神经元间循环的输入的迹，累积的内输入效应
            recurrent_input_trace (torch.Tensor): 在神经元间循环的输出的迹，累积的内输出效应
        """
        if multi_time_step:
            input_spike_train = input.clone()
            time_steps = input.shape[0]
            batch_size = input.shape[1]
        else:
            input_spike_train = input[None]
            time_steps = 1
            batch_size = input.shape[0]
        output, output_last = forward_func(input, output_0)
        if multi_time_step:
            output_spike_train = output.clone()
        else:
            output_spike_train = output[None]
        recurrent_spike_train = torch.zeros_like(output_spike_train)
        recurrent_spike_train[0] = output_0
        recurrent_spike_train[1:] = output_spike_train[:-1]
        delta_input_weight = torch.zeros_like(weight)
        delta_recurrent_weight = torch.zeros_like(recurrent_weight)
        if training:
            for t in range(time_steps):
                delta_recurrent_weight, recurrent_input_trace, recurrent_output_trace = stdp_online(
                    delta_weight = delta_recurrent_weight,
                    input_trace = recurrent_input_trace,
                    output_trace = recurrent_output_trace,
                    input_spike_train = recurrent_spike_train[t],
                    output_spike_train = output_spike_train[t],
                    a_pos = a_pos,
                    tau_pos = tau_pos,
                    a_neg = a_neg,
                    tau_neg = tau_neg
                )
                delta_input_weight, input_trace, output_trace = stdp_online(
                    delta_weight = delta_input_weight,
                    input_trace = input_trace,
                    output_trace = output_trace,
                    input_spike_train = input_spike_train[t],
                    output_spike_train = output_spike_train[t],
                    a_pos = a_pos,
                    tau_pos = tau_pos,
                    a_neg = a_neg,
                    tau_neg = tau_neg
                )
        ctx.save_for_backward(delta_input_weight, delta_recurrent_weight, input)
        return output, output_last, input_trace, output_trace, recurrent_input_trace, recurrent_output_trace


    @staticmethod
    def backward(ctx: torch.Any, grad_output: torch.Tensor, grad_output_last: torch.Tensor, grad_input_trace: torch.Tensor, grad_output_trace: torch.Tensor, grad_recurrent_input_trace: torch.Tensor, grad_recurrent_output_trace: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None, None]:
        """
        利用STDP进行学习的液体状态机的反向传播函数。
        Args:
            ctx (torch.Any): 上下文
            grad_output (torch.Tensor): 输出脉冲序列梯度
            grad_output_last (torch.Tensor): 最终的循环脉冲梯度
            grad_input_trace (torch.Tensor): 输入的迹梯度
            grad_output_trace (torch.Tensor): 输出的迹梯度
            grad_recurrent_input_trace (torch.Tensor): 在神经元间循环的输入的迹梯度
            grad_recurrent_input_trace (torch.Tensor): 在神经元间循环的输出的迹梯度
        Returns:
            grad_input (torch.Tensor): 输入脉冲序列梯度
            grad_output_0 (torch.Tensor): 初始状态下的循环脉冲梯度
            grad_weight (torch.Tensor): 权重矩阵梯度
            grad_input_trace (torch.Tensor): 输入的迹梯度
            grad_output_trace (torch.Tensor): 输出的迹梯度
            grad_recurrent_weight (torch.Tensor): 循环权重矩阵梯度
            grad_recurrent_input_trace (torch.Tensor): 在神经元间循环的输入的迹梯度
            grad_recurrent_input_trace (torch.Tensor): 在神经元间循环的输出的迹梯度
            grad_forward_func (None): 前向传播函数的梯度，为None
            grad_a_pos (None): STDP参数A+的梯度，为None
            grad_tau_pos (None): STDP参数tau+的梯度，为None
            grad_a_neg (None): STDP参数A-的梯度，为None
            grad_tau_neg (None): STDP参数tau-的梯度，为None
            grad_training (None): 是否正在训练的梯度，为None
            grad_multi_time_step (None): 是否为多时间步模式的梯度，为None
        """
        delta_input_weight, delta_recurrent_weight, input = ctx.saved_tensors
        delta_input_weight = -delta_input_weight
        delta_recurrent_weight = -delta_recurrent_weight
        return torch.zeros_like(input), torch.zeros_like(grad_output_last), delta_input_weight, torch.zeros_like(grad_input_trace), torch.zeros_like(grad_output_trace), delta_recurrent_weight, torch.zeros_like(grad_recurrent_input_trace), torch.zeros_like(grad_recurrent_output_trace), None, None, None, None, None, None, None


class STDPLSM(LSM):
    def __init__(self, adjacent: torch.Tensor, soma: Module, a_pos: float = 1.0, tau_pos: float = 2.0, a_neg: float = 1.0, tau_neg: float = 2.0, multi_time_step: bool = True, reset_after_process: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        使用STDP学习机制时的全连接层。
        Args:
            adjacent (torch.Tensor): 邻接矩阵，行为各个神经元的轴突，列为各个神经元的树突，1为有从轴突指向树突的连接，0为没有
            soma (nn.Module): 使用的脉冲神经元胞体，在matterhorn_pytorch.snn.soma中选择
            a_pos (float): STDP参数A+
            tau_pos (float): STDP参数tau+
            a_neg (float): STDP参数A-
            tau_neg (float): STDP参数tau-
            multi_time_step (bool): 是否调整为多个时间步模式
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        self.input_trace = None
        self.output_trace = None
        self.recurrent_input_trace = None
        self.recurrent_output_trace = None
        super().__init__(
            adjacent = adjacent,
            soma = soma,
            multi_time_step = multi_time_step,
            reset_after_process = reset_after_process,
            device = device,
            dtype = dtype
        )
        self.a_pos = a_pos
        self.tau_pos = tau_pos
        self.a_neg = a_neg
        self.tau_neg = tau_neg
        self.reset()


    def reset(self) -> nn.Module:
        """
        重置模型。
        """
        self.input_trace = SF.reset_tensor(self.input_trace, 0.0)
        self.output_trace = SF.reset_tensor(self.output_trace, 0.0)
        self.recurrent_input_trace = SF.reset_tensor(self.recurrent_input_trace, 0.0)
        self.recurrent_output_trace = SF.reset_tensor(self.recurrent_output_trace, 0.0)
        return super().reset()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 当前输入，形状为[B, I]（单步）或[T, B, I]（多步）
        Returns:
            y (torch.Tensor): 当前输出，形状为[B, O]（单步）或[T, B, O]（多步）
        """
        if self.multi_time_step:
            time_steps = x.shape[0]
            batch_size = x.shape[1]
            self.y_0 = SF.init_tensor(self.y_0, x[0])
        else:
            time_steps = 1
            batch_size = x.shape[0]
            self.y_0 = SF.init_tensor(self.y_0, x)
        trace_shape = torch.zeros_like(self.input_weight)[None].repeat_interleave(batch_size, dim = 0)
        self.input_trace = SF.init_tensor(self.input_trace, trace_shape)
        self.output_trace = SF.init_tensor(self.output_trace, trace_shape)
        recurrent_trace_shape = torch.zeros_like(self.recurrent_weight)[None].repeat_interleave(batch_size, dim = 0)
        self.recurrent_input_trace = SF.init_tensor(self.recurrent_input_trace, recurrent_trace_shape)
        self.recurrent_output_trace = SF.init_tensor(self.recurrent_output_trace, recurrent_trace_shape)
        y, self.y_0, self.input_trace, self.output_trace, self.recurrent_input_trace, self.recurrent_output_trace = f_stdp_lsm.apply(x, self.y_0, self.input_weight, self.input_trace, self.output_trace, self.recurrent_weight, self.recurrent_input_trace, self.recurrent_output_trace, self.forward_multi_time_step if self.multi_time_step else self.forward_single_time_step, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg, self.training, self.multi_time_step)
        return y