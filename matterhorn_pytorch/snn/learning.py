# -*- coding: UTF-8 -*-
"""
自定义SNN的学习机制，以STDP作为样例。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.snn.skeleton import Module
from typing import Union
try:
    from rich import print
except:
    pass


@torch.jit.script
def stdp_py(delta_weight: torch.Tensor, input_shape: int, output_shape: int, time_steps: int, input_spike_train: torch.Tensor, output_spike_train: torch.Tensor, a_pos: float, tau_pos: float, a_neg: float, tau_neg: float) -> torch.Tensor:
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
            for ti in range(time_steps):
                if not output_spike_train[ti, i]:
                    continue
                for tj in range(time_steps):
                    if not input_spike_train[tj, j]:
                        continue
                    dt = ti - tj
                    if dt > 0:
                        weight += a_pos * torch.exp(-dt / tau_pos)
                    else:
                        weight += -a_neg * torch.exp(dt / tau_neg)
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


def stdp(delta_weight: torch.Tensor, input_spike_train: torch.Tensor, output_spike_train: torch.Tensor, a_pos: float, tau_pos: float, a_neg: float, tau_neg: float) -> torch.Tensor:
    """
    STDP总函数，视情况调用函数
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
    """
    input_shape = delta_weight.shape[1]
    output_shape = delta_weight.shape[0]
    assert input_spike_train.shape[1] == output_spike_train.shape[1] and input_spike_train.shape[0] == input_shape and output_spike_train.shape[0] == output_shape, "The shape of tensors is not compatible: weight (o=%d, i=%d) with input (i=%d, t=%d) and output (o=%d, t=%d)" % (delta_weight.shape[0], delta_weight.shape[1], input_spike_train.shape[0], input_spike_train.shape[1], output_spike_train.shape[0], output_spike_train.shape[1])
    time_steps = input_spike_train.shape[1]
    w_type = delta_weight.device.type
    w_idx = delta_weight.device.index
    i_type = input_spike_train.device.type
    i_idx = input_spike_train.device.index
    o_type = output_spike_train.device.type
    o_idx = output_spike_train.device.index
    assert (w_type == i_type and i_type == o_type) and (w_idx == i_idx and i_idx == o_idx), "The type of weight matrix, input spike train and output spike train should be the same."
    device_type = w_type
    device_idx = w_idx
    if device_type == "cuda" and stdp_cuda is not None:
        stdp_cuda(delta_weight, input_shape, output_shape, time_steps, input_spike_train, output_spike_train, a_pos, tau_pos, a_neg, tau_neg)
        return delta_weight
    if stdp_cpp is not None:
        delta_weight_cpu = delta_weight.cpu()
        stdp_cpp(delta_weight_cpu, input_shape, output_shape, time_steps, input_spike_train.cpu(), output_spike_train.cpu(), a_pos, tau_pos, a_neg, tau_neg)
        delta_weight = delta_weight_cpu.to(delta_weight)
        return delta_weight
    return stdp_py(delta_weight, input_shape, output_shape, time_steps, input_spike_train, output_spike_train, a_pos, tau_pos, a_neg, tau_neg)


class STDPLinear(Module, nn.Linear):
    def __init__(self, in_features: int, out_features: int, soma: nn.Module, a_pos: float = 0.05, tau_pos: float = 2.0, a_neg: float = 0.05, tau_neg: float = 2.0, lr: float = 0.01, multi_time_step: bool = True, device = None, dtype = None) -> None:
        """
        使用STDP学习机制时的全连接层
        Args:
            in_features (int): 输入长度，用法同nn.Linear
            out_features (int): 输出长度，用法同nn.Linear
            soma (nn.Module): 使用的脉冲神经元胞体，在matterhorn_pytorch.snn.soma中选择
            a_pos (float): STDP参数A+
            tau_pos (float): STDP参数tau+
            a_neg (float): STDP参数A-
            tau_neg (float): STDP参数tau-
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Module.__init__(
            self,
            multi_time_step = multi_time_step,
            reset_after_process = False
        )
        nn.Linear.__init__(
            self,
            in_features = in_features, 
            out_features = out_features,
            bias = False,
            device = device,
            dtype = dtype
        )
        self.weight.requires_grad_(False)
        self.soma = soma
        self.a_pos = a_pos
        self.tau_pos = tau_pos
        self.a_neg = a_neg
        self.tau_neg = tau_neg
        self.lr = lr
        self.reset()


    def reset(self) -> None:
        """
        重置整个神经元。
        """
        self.input_spike_seq = []
        self.output_spike_seq = []
        is_snn_module = isinstance(self.soma, Module)
        if is_snn_module:
            self.soma.reset()


    def train(self, mode: Union[str, bool] = "stdp") -> None:
        """
        切换训练和测试模式。
        Args:
            mode (str | bool): 采用何种训练方式，None为测试模式
        """
        if isinstance(mode, str):
            mode = mode.lower()
        is_snn_module = isinstance(self.soma, Module)
        if is_snn_module:
            self.soma.train(mode)
        else:
            self.soma.train(mode in (True, "bp"))
    

    def eval(self) -> None:
        """
        切换测试模式。
        """
        self.soma.eval()


    def step(self) -> None:
        """
        对整个神经元应用STDP使其更新。
        """
        time_steps = len(self.input_spike_seq)
        if self.multi_time_step:
            input_spike_train = torch.cat(self.input_spike_seq)
            output_spike_train = torch.cat(self.output_spike_seq)
        else:
            input_spike_train = torch.stack(self.input_spike_seq)
            output_spike_train = torch.stack(self.output_spike_seq)
        if len(input_spike_train.shape) == 3: # [B, T, L]
            batch_size = input_spike_train.shape[1]
            for b in range(batch_size):
                delta_weight = torch.zeros_like(self.weight)
                delta_weight = stdp(delta_weight, self.in_features, self.out_features, time_steps, input_spike_train[:, b], output_spike_train[:, b], self.a_pos, self.tau_pos, self.a_neg, self.tau_neg)
                self.weight += self.lr * delta_weight
        else: # [T, L]
            delta_weight = torch.zeros_like(self.weight)
            delta_weight = stdp(delta_weight, self.in_features, self.out_features, time_steps, input_spike_train, output_spike_train, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg)
            self.weight += self.lr * delta_weight


    def forward_single_time_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。True
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$，形状为[B, I]
        Returns:
            o (torch.Tensor): 当前层的输出脉冲$O_{i}^{l}(t)$，形状为[B, O]
        """
        self.input_spike_seq.append(o.clone().detach())
        x = nn.Linear.forward(self, o)
        o = self.soma(x)
        self.output_spike_seq.append(o.clone().detach())
        return o


    def forward_multi_time_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}$，形状为[T, B, I]
        Returns:
            o (torch.Tensor): 当前层的输出脉冲$O_{i}^{l}$，形状为[T, B, O]
        """
        self.input_spike_seq.append(o.clone().detach())
        time_steps = o.shape[0]
        batch_size = o.shape[1]
        o = o.flatten(0, 1)
        x = nn.Linear.forward(o)
        output_shape = [time_steps, batch_size] + list(x.shape[1:])
        x = x.reshape(output_shape)
        o = self.soma(x)
        self.output_spike_seq.append(o.clone().detach())
        return o


    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            o (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        if self.multi_time_step:
            o = self.forward_multi_time_step(o)
        else:
            o = self.forward_single_time_step(o)
        return o