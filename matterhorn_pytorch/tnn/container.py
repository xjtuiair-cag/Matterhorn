# -*- coding: UTF-8 -*-
"""
TNN 网络容器，用于串联多个认知柱或自定义模块。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.snn.skeleton import Module as _Module
from matterhorn_pytorch.tnn.cognitive import CognitiveColumn
from typing import Any as _Any, Tuple as _Tuple, Union as _Union, Optional as _Optional, Iterable as _Iterable


class Sequential(_Module, nn.Sequential):
    """
    TNN 序列容器，支持认知柱及任意 nn.Module 串联。
    """

    def __init__(self, *args: _Tuple[nn.Module], return_states: bool = False) -> None:
        """
        Args:
            *args (nn.Module): 按顺序排列的模块
            return_states (bool): 是否返回各层中间状态
        """
        _Module.__init__(self)
        nn.Sequential.__init__(self, *args)
        self.return_states = return_states


    def extra_repr(self) -> str:
        return "return_states=%s" % (self.return_states,)


    def forward(self, x: torch.Tensor, **kwargs) -> _Union[torch.Tensor, _Tuple[torch.Tensor, _Iterable]]:
        """
        前向传播。若子模块为 CognitiveColumn，自动传递 merged_input。
        Args:
            x (torch.Tensor): 输入时间簇 [B, M]
        Returns:
            y (torch.Tensor): 输出时间簇
            states (list | None): 各层状态（当 return_states=True）
        """
        states = []
        for module in self:
            if isinstance(module, CognitiveColumn):
                y, st = module(merged_input = x)
            elif hasattr(module, "forward") and "merged_input" in module.forward.__code__.co_varnames:
                y, st = module(merged_input = x)
            else:
                out = module(x)
                if isinstance(out, _Tuple):
                    y, st = out[0], out[1] if len(out) > 1 else None
                else:
                    y, st = out, None
            x = y
            if self.return_states:
                states.append(st)
        if self.return_states:
            return x, states
        return x
