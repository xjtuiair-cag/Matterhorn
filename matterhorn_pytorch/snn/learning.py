# -*- coding: UTF-8 -*-
"""
自定义SNN的学习机制，以STDP作为样例。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.snn.skeleton import Module
try:
    from rich import print
except:
    pass


@torch.jit.script
def stdp_py(delta_weight: torch.Tensor, input_shape: int, output_shape: int, time_steps: int, input_spike_train: torch.Tensor, output_spike_train: torch.Tensor, a_pos: float, tau_pos: float, a_neg: float, tau_neg: float) -> torch.Tensor:
    """
    STDP的python版本实现，不到万不得已不会调用（性能是灾难级别的）
    @params:
        delta_weight: torch.Tensor 权重矩阵，形状为[output_shape, input_shape]
        input_shape: int 输入长度
        output_shape: int 输出长度
        time_steps: int 时间步长
        input_spike_train: torch.Tensor 输入脉冲序列，形状为[input_shape, time_steps]
        output_spike_train: torch.Tensor 输出脉冲序列，形状为[output_shape, time_steps]
        a_pos: float STDP参数A+
        tau_pos: float STDP参数tau+
        a_neg: float STDP参数A-
        tau_neg: float STDP参数tau-
    @return:
        delta_weight: torch.Tensor 权重增量
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


def stdp(delta_weight: torch.Tensor, input_shape: int, output_shape: int, time_steps: int, input_spike_train: torch.Tensor, output_spike_train: torch.Tensor, a_pos: float, tau_pos: float, a_neg: float, tau_neg: float) -> torch.Tensor:
    """
    STDP总函数，视情况调用函数
    @params:
        delta_weight: torch.Tensor 权重矩阵，形状为[output_shape, input_shape]
        input_shape: int 输入长度
        output_shape: int 输出长度
        time_steps: int 时间步长
        input_spike_train: torch.Tensor 输入脉冲序列，形状为[input_shape, time_steps]
        output_spike_train: torch.Tensor 输出脉冲序列，形状为[output_shape, time_steps]
        a_pos: float STDP参数A+
        tau_pos: float STDP参数tau+
        a_neg: float STDP参数A-
        tau_neg: float STDP参数tau-
    """
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
    def __init__(self, in_features: int, out_features: int, soma: nn.Module, a_pos: float = 0.05, tau_pos: float = 2.0, a_neg: float = 0.05, tau_neg: float = 2.0, lr: float = 0.01, device = None, dtype = None) -> None:
        """
        使用STDP学习机制时的全连接层
        @params:
            in_features: int 输入长度，用法同nn.Linear
            out_features: int 输出长度，用法同nn.Linear
            soma: nn.Module 使用的脉冲神经元胞体，在matterhorn_pytorch.snn.soma中选择
            a_pos: float STDP参数A+
            tau_pos: float STDP参数tau+
            a_neg: float STDP参数A-
            tau_neg: float STDP参数tau-
        """
        Module.__init__(self)
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
    

    def start_step(self) -> None:
        """
        开始训练
        """
        is_snn_module = isinstance(self.soma, Module)
        if is_snn_module:
            self.soma.start_step()
    

    def stop_step(self) -> None:
        """
        停止训练
        """
        is_snn_module = isinstance(self.soma, Module)
        if is_snn_module:
            self.soma.stop_step()


    def reset(self) -> None:
        """
        重置整个神经元
        """
        self.input_spike_seq = []
        self.output_spike_seq = []
        is_snn_module = isinstance(self.soma, Module)
        if is_snn_module:
            self.soma.reset()


    def step_once(self) -> None:
        """
        对整个神经元应用STDP使其更新
        """
        time_steps = len(self.input_spike_seq)
        input_spike_train = torch.stack(self.input_spike_seq)
        output_spike_train = torch.stack(self.output_spike_seq)
        if len(input_spike_train.shape) == 3:
            batch_size = input_spike_train.shape[1]
            for b in range(batch_size):
                delta_weight = torch.zeros_like(self.weight)
                delta_weight = stdp(delta_weight, self.in_features, self.out_features, time_steps, input_spike_train[:, b], output_spike_train[:, b], self.a_pos, self.tau_pos, self.a_neg, self.tau_neg)
                self.weight += self.lr * delta_weight
        else:
            delta_weight = torch.zeros_like(self.weight)
            delta_weight = stdp(delta_weight, self.in_features, self.out_features, time_steps, input_spike_train, output_spike_train, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg)
            self.weight += self.lr * delta_weight
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            x: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$
        @return:
            o: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$
        """
        self.input_spike_seq.append(x.clone().detach().requires_grad_(True))
        x = super().forward(x)
        x = self.soma(x)
        self.output_spike_seq.append(x.clone().detach().requires_grad_(True))
        return x