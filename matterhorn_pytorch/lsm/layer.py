# -*- coding: UTF-8 -*-
"""
液体状态机。
使用邻接矩阵表示神经元的连接方向。
"""


import torch
import torch.nn as nn
import torch.nn.functional as _F
from typing import Any as _Any, Tuple as _Tuple, Callable as _Callable
import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.skeleton import Module as _Module
from matterhorn_pytorch.snn.soma import Soma as _Soma
from matterhorn_pytorch.snn.container import Temporal as _Temporal
from matterhorn_pytorch.training.functional import stdp_online as _stdp_online


class LSM(_Module):
    def __init__(self, adjacent: torch.Tensor, soma: _Soma, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        液体状态机。
        Args:
            adjacent (torch.Tensor): 邻接矩阵，行为各个神经元的轴突，列为各个神经元的树突，1为有从轴突指向树突的连接，0为没有
            soma (nn.Module): 使用的脉冲神经元胞体，在matterhorn_pytorch.snn.soma中选择
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        assert len(adjacent.shape) == 2 and adjacent.shape[0] == adjacent.shape[1], "Incorrect adjacent matrix."
        super().__init__()
        self.o = None
        self.adjacent = nn.Parameter(adjacent.to(torch.float), requires_grad = False)
        self.soma = soma
        self.neuron_num = self.adjacent.shape[0]
        self.weight = nn.Parameter(torch.empty((self.neuron_num, self.neuron_num), device = device, dtype = dtype))
        nn.init.kaiming_uniform_(self.weight, a = 5.0 ** 0.5)


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["neuron_num=%d" % self.neuron_num, "adjacent=\n%s\n" % self.adjacent]) + ((", " + super().extra_repr()) if len(super().extra_repr()) else "")


    def reset(self) -> _Module:
        """
        重置整个神经元。
        """
        if isinstance(self.soma, _Module):
            self.soma.reset()
        self.o = _SF.reset_tensor(self.o, 0.0)
        return super().reset()


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 来自突触的输入电位$X_{i}^{l}(t)$
        Returns:
            o (torch.Tensor): 胞体当前的输出脉冲$O_{i}^{l}(t)$
        """
        self.o = _SF.init_tensor(self.o, x)
        y = x + _F.linear(self.o, self.weight * self.adjacent.T, None)
        o = self.soma.forward_step(y)
        self.o = o
        return o