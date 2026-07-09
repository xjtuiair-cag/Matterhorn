# -*- coding: UTF-8 -*-
"""
认知柱（Cognitive Column, CC）。

Smith 书第7.2节：CC = 兴奋柱(EC) + 前馈抑制柱(FFI) + 侧向抑制柱(LI)。
处理流程：合并输入 → FFI → EC → LI → 输出脉冲簇。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.snn.skeleton import Module as _Module
from matterhorn_pytorch.tnn.column import ExcitatoryColumn
from matterhorn_pytorch.tnn.inhibition import FeedforwardInhibition, LateralInhibition
from matterhorn_pytorch.tnn import volley as _TV
from typing import Tuple as _Tuple, Optional as _Optional, Iterable as _Iterable


class CognitiveColumn(_Module):
    """
    认知柱：由 FFI、EC、LI 组成的基本 TNN 计算单元。
    """

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int,
        inputs_per_neuron: _Optional[int] = None,
        connectivity: _Optional[torch.Tensor] = None,
        t_f: float = 1.3,
        k_f: int = 10,
        t_l: float = 5.0,
        k_l: int = 3,
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
            num_inputs (int): 输入线数（合并后 bundle 宽度）
            num_neurons (int): 兴奋柱神经元数
            inputs_per_neuron (int | None): 片内连接数
            connectivity (torch.Tensor | None): 自定义连接掩码
            t_f, k_f: FFI 参数
            t_l, k_l: LI 参数
            其余参数: 兴奋性神经元参数
            seed (int | None): 随机连接种子
            device, dtype: 张量工厂参数
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.ffi = FeedforwardInhibition(t_f = t_f, k_f = k_f)
        self.ec = ExcitatoryColumn(
            num_inputs = num_inputs,
            num_neurons = num_neurons,
            inputs_per_neuron = inputs_per_neuron,
            connectivity = connectivity,
            w_max = w_max,
            t_max = t_max,
            tr = tr,
            tf = tf,
            theta = theta,
            num_steps = num_steps,
            seed = seed,
            device = device,
            dtype = dtype
        )
        self.li = LateralInhibition(t_l = t_l, k_l = k_l)


    def extra_repr(self) -> str:
        return "num_inputs=%d, num_neurons=%d" % (self.num_inputs, self.num_neurons)


    def forward(
        self,
        spike_times: _Iterable[torch.Tensor] = None,
        merged_input: _Optional[torch.Tensor] = None
    ) -> _Tuple[torch.Tensor, dict]:
        """
        认知柱前向传播。
        Args:
            spike_times (Iterable[torch.Tensor] | None): 待合并的多个输入时间簇
            merged_input (torch.Tensor | None): 已合并的输入时间簇 [B, I]
        Returns:
            output (torch.Tensor): 输出时间簇 [B, N]
            states (dict): 中间状态（ffi/ec/li 各阶段输出）
        """
        if merged_input is None:
            if spike_times is None:
                raise ValueError("Either spike_times or merged_input must be provided.")
            merged_input = _TV.merge_time_volleys(*spike_times)

        t_norm, t_min = _TV.normalize_time_volley(merged_input)
        ffi_out = self.ffi(t_norm, normalized = True)
        ec_out, _ = self.ec(ffi_out, global_times = False)
        li_out = self.li(ec_out, normalized = True)
        output = _TV.denormalize_time_volley(li_out, t_min)

        states = {
            "merged_input": merged_input,
            "ffi_out": ffi_out,
            "ec_out": ec_out,
            "li_out": li_out,
            "output": output
        }
        return output, states
