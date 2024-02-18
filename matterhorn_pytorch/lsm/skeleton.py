# -*- coding: UTF-8 -*-
"""
液体状态机。
使用邻接矩阵表示神经元的连接方向。
"""


import torch
import torch.nn as nn
import math
from matterhorn_pytorch import snn
from matterhorn_pytorch.snn.container import Temporal
from matterhorn_pytorch.snn.skeleton import Module
from matterhorn_pytorch.training.functional import stdp
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
            soma (snn.Module): 胞体
            multi_time_step (bool): 是否调整为多个时间步模式
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__(
            multi_time_step = multi_time_step,
            reset_after_process = reset_after_process
        )
        assert len(adjacent.shape) == 2 and adjacent.shape[0] == adjacent.shape[1], "Incorrect adjacent matrix."
        self.adjacent = nn.Parameter(adjacent.to(torch.float), requires_grad = False)
        self.neuron_num = self.adjacent.shape[0]
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
        self.weight = nn.Parameter(torch.empty((self.neuron_num, self.neuron_num), device = device, dtype = dtype), requires_grad = True)
        nn.init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        self.weight_input = nn.Parameter(torch.empty((self.neuron_num), device = device, dtype = dtype), requires_grad = True)
        nn.init.normal_(self.weight_input)
        self.reset()


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return "neuron_num=%d, adjacent=\n%s\n, multi_time_step=%s" % (self.neuron_num, self.adjacent, str(self.multi_time_step))


    def init_tensor(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        校正整个输入形状。
        Args:
            u (torch.Tensor): 待校正的输入，可能是张量或浮点值
            x (torch.Tensor): 带有正确数据类型、所在设备和形状的张量
        Returns:
            u (torch.Tensor): 经过校正的输入张量
        """
        if isinstance(u, float):
            u = torch.full_like(x, u)
            u = u.detach().requires_grad_(True)
        return u


    def permute_t_b(self, x: torch.Tensor) -> torch.Tensor:
        """
        将时间步与批大小对调。
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            y (torch.Tensor): 输出张量
        """
        idx = [i for i, j in enumerate(x.shape)]
        idx[0], idx[1] = idx[1], idx[0]
        y = x.permute(*idx)
        return y


    def reset(self) -> Module:
        """
        重置整个神经元。
        """
        if isinstance(self.soma, snn.Module):
            self.soma.reset()
        self.last_output = 0.0
        return super().reset()


    def f_synapse(self, y_0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_0 = self.init_tensor(y_0, x)
        y = nn.functional.linear(y_0, self.weight * self.adjacent.T, None) + (x * self.weight_input)
        return y


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 当前输入，形状为[B, I]
        Returns:
            y (torch.Tensor): 当前输出，形状为[B, O]
        """
        x = self.f_synapse(self.last_output, x)
        y = self.soma(x)
        self.last_output = y.clone()
        return y


    def forward_multi_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 输入序列，形状为[T, B, I]
        Returns:
            y (torch.Tensor): 输出序列，形状为[T, B, O]
        """
        time_steps = x.shape[0]
        y_seq = []
        for t in range(time_steps):
            y_seq.append(self.forward_single_time_step(x[t]))
        y = torch.stack(y_seq)
        return y


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 当前输入，形状为[B, I]（单步）或[B, T, I]（多步）
        Returns:
            y (torch.Tensor): 当前输出，形状为[B, O]（单步）或[B, T, O]（多步）
        """
        if self.multi_time_step:
            x = self.permute_t_b(x)
            y = self.forward_multi_time_step(x)
            y = self.permute_t_b(y)
            if self.reset_after_process:
                self.reset()
        else:
            y = self.forward_single_time_step(x)
        return y


class STDPLSM(LSM):
    def __init__(self, adjacent: torch.Tensor, soma: Module, a_pos: float = 0.05, tau_pos: float = 2.0, a_neg: float = 0.05, tau_neg: float = 2.0, lr: float = 0.01, multi_time_step: bool = True, reset_after_process: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        super().__init__(
            adjacent = adjacent,
            soma = soma,
            multi_time_step = multi_time_step,
            reset_after_process = reset_after_process,
            device = device,
            dtype = dtype
        )
        self.input_spike_seq = []
        self.output_spike_seq = []
        self.weight.requires_grad_(False)
        self.weight_input.requires_grad_(False)
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
        self.a_pos = a_pos
        self.tau_pos = tau_pos
        self.a_neg = a_neg
        self.tau_neg = tau_neg
        self.lr = lr


    def step(self, *args, **kwargs) -> Module:
        """
        对整个神经元应用STDP使其更新。
        """
        if not self.training:
            return
        if self.multi_time_step:
            input_spike_train = torch.cat(self.input_spike_seq)
            output_spike_train = torch.cat(self.output_spike_seq)
        else:
            input_spike_train = torch.stack(self.input_spike_seq)
            output_spike_train = torch.stack(self.output_spike_seq)
        is_batched = input_spike_train.ndim >= 3
        if is_batched:
            input_spike_train = input_spike_train.permute(1, 0, 2)
            output_spike_train = output_spike_train.permute(1, 0, 2)
        else:
            input_spike_train = input_spike_train.reshape(input_spike_train.shape[0], 1, input_spike_train.shape[1])
            output_spike_train = output_spike_train.reshape(output_spike_train.shape[0], 1, output_spike_train.shape[1])
        # 将不同维度的输入与输出张量形状统一转为[T, B, L]
        recurrent_spike_train = torch.zeros_like(output_spike_train)
        recurrent_spike_train[1:] = output_spike_train[:-1]
        delta_weight = torch.zeros_like(self.weight)
        delta_weight = stdp(delta_weight, recurrent_spike_train, output_spike_train, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg)
        self.weight += self.lr * delta_weight
        delta_weight_input = torch.zeros_like(self.weight)
        delta_weight_input = stdp(delta_weight_input, input_spike_train, output_spike_train, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg)
        self.weight_input += self.lr * torch.diagonal(delta_weight_input)
        self.input_spike_seq = []
        self.output_spike_seq = []
        return super().step(*args, **kwargs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 当前输入，形状为[B, I]（单步）或[B, T, I]（多步）
        Returns:
            y (torch.Tensor): 当前输出，形状为[B, O]（单步）或[B, T, O]（多步）
        """
        if self.multi_time_step:
            x = self.permute_t_b(x)
            y = self.forward_multi_time_step(x)
            if self.training:
                self.input_spike_seq.append(x.clone().detach())
                self.output_spike_seq.append(y.clone().detach())
            y = self.permute_t_b(y)
            if self.reset_after_process:
                self.reset()
        else:
            y = self.forward_single_time_step(x)
            if self.training:
                self.input_spike_seq.append(x.clone().detach())
                self.output_spike_seq.append(y.clone().detach())
        return y