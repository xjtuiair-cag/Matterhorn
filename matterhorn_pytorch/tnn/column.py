# -*- coding: UTF-8 -*-
"""
兴奋柱（Excitatory Column, EC）。

Smith 书第7.4节：并行兴奋性神经元阵列，对输入脉冲簇做局部归一化、
独立评估、再合并为输出脉冲簇。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.snn.skeleton import Module as _Module
from matterhorn_pytorch.tnn.neuron import ExcitatoryNeuron
from matterhorn_pytorch.tnn import volley as _TV
from typing import Tuple as _Tuple, Optional as _Optional


def build_random_connectivity(
    num_inputs: int,
    num_neurons: int,
    inputs_per_neuron: int,
    generator: _Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    为兴奋柱构建伪随机片内连接（Smith 书第9.5节 50×25×24 网络）。
    Args:
        num_inputs (int): 输入线数
        num_neurons (int): 神经元数
        inputs_per_neuron (int): 每个神经元连接的输入线数
        generator (torch.Generator | None): 随机数生成器
    Returns:
        mask (torch.Tensor): 连接掩码 [num_neurons, num_inputs]
    """
    mask = torch.zeros(num_neurons, num_inputs)
    k = min(inputs_per_neuron, num_inputs)
    for n in range(num_neurons):
        perm = torch.randperm(num_inputs, generator = generator)[:k]
        mask[n, perm] = 1.0
    return mask


class ExcitatoryColumn(_Module):
    """
    兴奋柱：由多个并行兴奋性神经元组成。
    """

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int,
        inputs_per_neuron: _Optional[int] = None,
        connectivity: _Optional[torch.Tensor] = None,
        w_max: float = 4.0,
        t_max: float = 16.0,
        tr: float = 10.0,
        tf: float = 40.0,
        theta: float = 19.0,
        num_steps: int = 256,
        seed: _Optional[int] = None,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        """
        Args:
            num_inputs (int): 输入线数
            num_neurons (int): 柱内兴奋性神经元数
            inputs_per_neuron (int | None): 每神经元连接数，默认 num_inputs//2
            connectivity (torch.Tensor | None): 自定义连接掩码 [N, I]
            w_max, t_max, tr, tf, theta, num_steps: 神经元参数
            seed (int | None): 随机连接种子
            device (torch.device): 设备
            dtype (torch.dtype): 数据类型
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        factory = {"device": device, "dtype": dtype}

        if connectivity is None:
            k = inputs_per_neuron if inputs_per_neuron is not None else max(1, num_inputs // 2)
            gen = torch.Generator()
            if seed is not None:
                gen.manual_seed(seed)
            connectivity = build_random_connectivity(num_inputs, num_neurons, k, gen)
        self.register_buffer("connectivity", connectivity.to(**factory))

        self.neurons = nn.ModuleList([
            ExcitatoryNeuron(
                num_inputs = num_inputs,
                w_max = w_max,
                t_max = t_max,
                tr = tr,
                tf = tf,
                theta = theta,
                num_steps = num_steps,
                connectivity = self.connectivity[n],
                device = device,
                dtype = dtype
            )
            for n in range(num_neurons)
        ])


    def extra_repr(self) -> str:
        return "num_inputs=%d, num_neurons=%d" % (self.num_inputs, self.num_neurons)


    def forward(
        self,
        spike_times: torch.Tensor,
        global_times: bool = True
    ) -> _Tuple[torch.Tensor, torch.Tensor]:
        """
        兴奋柱前向传播。
        Args:
            spike_times (torch.Tensor): 输入时间簇 [B, I]
            global_times (bool): 输入/输出是否为全局时间参考系
        Returns:
            out_volley (torch.Tensor): 输出时间簇 [B, N]
            t_min (torch.Tensor): 输入簇最早脉冲时间 [B]
        """
        outputs = []
        t_min = None
        for neuron in self.neurons:
            out_t, t_ref = neuron(spike_times, global_times = global_times)
            outputs.append(out_t.unsqueeze(-1))
            if t_min is None:
                t_min = t_ref
        out_volley = torch.cat(outputs, dim = -1)
        return out_volley, t_min
