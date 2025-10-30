# -*- coding: UTF-8 -*-
"""
脉冲神经网络的容器，用来容纳时间和空间维度的脉冲神经网络集合。
建议先在空间维度上构建完整的脉冲神经网络结构，再在多个时间步之内进行模拟。
"""


import torch
import torch.nn as nn
import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.skeleton import Module as _Module
from typing import Any as _Any, Iterable as _Iterable, Mapping as _Mapping, Tuple as _Tuple, Union as _Union, Optional as _Optional


class Container(_Module):
    def __init__(self) -> None:
        """
        容器的基类。
        """
        super().__init__()


class Sequential(Container, nn.Sequential):
    def __init__(self, *args: _Tuple[nn.Module], return_states: bool = True) -> None:
        """
        对Sequential进行重写，涵盖ANN与SNN的网络。
        Args:
            *args (*nn.Module): 按空间顺序传入的各个模块
        """
        Container.__init__(self)
        nn.Sequential.__init__(self, *args)
        self.return_states = return_states


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["return_states=%s" % (self.return_states,)])


    def forward(self, x: torch.Tensor, states: _Optional[_Iterable] = None) -> _Union[torch.Tensor, _Tuple[torch.Tensor, _Iterable]]:
        """
        前向传播函数，默认接受的张量形状为[B,...]
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            y (torch.Tensor): 输出张量
        """
        if states is None:
            states = [None] * len(self)
        assert len(states) == len(self), "States should be None or equal to modules' length."

        for idx, module in enumerate(self):
            args = module.forward.__code__.co_argcount - 1 # (减去self)
            if args <= 1:
                x = module(x)
            else:
                x = module(x, states[idx])
            if isinstance(x, _Tuple):
                x, states[idx] = x
        
        if self.return_states:
            return x, states
        return x