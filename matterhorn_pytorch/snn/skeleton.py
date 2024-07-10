# -*- coding: UTF-8 -*-
"""
SNN模块的框架，在torch.nn的基础上，定义了几个SNN的基本函数。
"""


import torch
import torch.nn as nn
from typing import Tuple as _Tuple


class Module(nn.Module):
    def __init__(self, multi_time_step: bool = False) -> None:
        """
        脉冲神经网络模块的骨架。
        Args:
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        nn.Module.__init__(self)
        self._multi_time_step = False
        if multi_time_step:
            self.multi_time_step_(multi_time_step)


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return super().extra_repr()


    @property
    def supports_single_time_step(self) -> bool:
        """
        是否支持单个时间步。
        Returns:
            if_support (bool): 是否支持单个时间步
        """
        return True


    @property
    def supports_multi_time_step(self) -> bool:
        """
        是否支持多个时间步。
        Returns:
            if_support (bool): 是否支持多个时间步
        """
        return True
    

    @property
    def multi_time_step(self) -> bool:
        """
        当前是否为多时间步模式。
        Returns:
            if_on (bool): 当前是否为多个时间步模式
        """
        return self._multi_time_step


    def multi_time_step_(self, if_on: bool) -> nn.Module:
        """
        调整模型的多时间步模式。
        Args
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
        """
        if self.supports_multi_time_step and if_on:
            self._multi_time_step = True
        elif self.supports_single_time_step and not if_on:
            self._multi_time_step = False
        for name, module in self.named_children():
            from matterhorn_pytorch.snn.container import Temporal as _Temporal
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                if if_on:
                    if module.supports_multi_time_step:
                        module.multi_time_step_(if_on)
                    else:
                        setattr(self, name, _Temporal(module))
                else:
                    if module.supports_single_time_step:
                        module.multi_time_step_(if_on)
                    elif isinstance(module, _Temporal):
                        setattr(self, name, module.module)
                    else:
                        raise ValueError("Unsupported time step conversion on module %s(%s)" % (name, module.__class__.__name__))
        return self


    def reset(self) -> nn.Module:
        """
        重置模型。
        """
        for module in self.children():
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.reset()
        return self


    def detach(self) -> nn.Module:
        """
        将模型中的某些变量从其计算图中分离。
        """
        for module in self.children():
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.detach()
        return self


    def forward_single_time_step(self, *args, **kwargs) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            *args: 输入
            **kwargs: 输入
        Returns:
            res (torch.Tensor): 输出
        """
        pass


    def forward_multi_time_steps(self, *args, **kwargs) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            *args: 输入
            **kwargs: 输入
        Returns:
            res (torch.Tensor): 输出
        """
        if not isinstance(args, _Tuple):
            args = (args,)
        time_steps = args[0].shape[0]
        result = []
        for t in range(time_steps):
            args_t = (x[t] if isinstance(x, torch.Tensor) else x for x in args)
            kwargs_t = {k: x[t] if isinstance(x, torch.Tensor) else x for k, x in kwargs}
            result.append(self.forward_single_time_step(*args_t, **kwargs_t))
        if isinstance(result[0], _Tuple):
            y = (torch.stack([result[t][col] for t in range(len(result))]) if isinstance(result[0][col], torch.Tensor) else result[0][col] for col in range(len(result[0])))
        else:
            y = torch.stack(result)
        return y


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            *args: 输入
            **kwargs: 输入
        Returns:
            res (torch.Tensor): 输出
        """
        if self.multi_time_step:
            res = self.forward_multi_time_steps(*args, **kwargs)
        else:
            res = self.forward_single_time_step(*args, **kwargs)
        return res