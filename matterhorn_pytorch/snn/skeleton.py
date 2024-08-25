# -*- coding: UTF-8 -*-
"""
SNN模块的框架，在torch.nn的基础上，定义了几个SNN的基本函数。
"""


import torch
import torch.nn as nn
from typing import Any as _Any, Tuple as _Tuple, Mapping as _Mapping


class Module(nn.Module):
    def __init__(self) -> None:
        """
        脉冲神经网络模块的骨架。
        """
        nn.Module.__init__(self)
        self._multi_step_mode = False


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ("step_mode=%s" % ('"m"' if self.multi_step_mode else '"s"')) if self.multi_step_mode else ""


    @property
    def supports_single_step_mode(self) -> bool:
        """
        是否支持单个时间步。
        Returns:
            if_support (bool): 是否支持单个时间步
        """
        return True


    @property
    def supports_multi_step_mode(self) -> bool:
        """
        是否支持多个时间步。
        Returns:
            if_support (bool): 是否支持多个时间步
        """
        return True


    @property
    def multi_step_mode(self) -> bool:
        """
        当前是否为多时间步模式。
        Returns:
            if_on (bool): 当前是否为多个时间步模式
        """
        return self._multi_step_mode
    

    @property
    def single_step_mode(self) -> bool:
        """
        当前是否为单时间步模式。
        Returns:
            if_on (bool): 当前是否为单个时间步模式
        """
        return not self._multi_step_mode


    def multi_step_mode_(self, if_on: bool = True, recursive: bool = True) -> nn.Module:
        """
        调整模型至多时间步模式。
        Args:
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
            recursive (bool): 是否递归调整子模块的时间步模式
        """
        if recursive:
            _ = [module.multi_step_mode_(if_on, recursive = recursive) if isinstance(module, Module) else None for module in self.children()]
        if self.supports_multi_step_mode and if_on:
            self._multi_step_mode = True
        elif self.supports_single_step_mode and not if_on:
            self._multi_step_mode = False
        else:
            raise ValueError("Unsupported time step conversion on module %s" % (self.__class__.__name__,))
        self.reset()
        return self
    

    def single_step_mode_(self, if_on: bool = True, recursive: bool = True) -> nn.Module:
        """
        调整模型至单时间步模式。
        Args
            if_on (bool): 当前需要调整为什么模式（True为单时间步模式，False为多时间步模式）
        """
        return self.multi_step_mode_(not if_on, recursive = recursive)


    def detach(self) -> nn.Module:
        """
        将模型中的某些变量从其计算图中分离。
        """
        _ = [module.detach() if isinstance(module, Module) else None for name, module in self.named_children()]
        return self


    def reset(self) -> nn.Module:
        """
        重置模型。
        """
        _ = [module.reset() if isinstance(module, Module) else None for name, module in self.named_children()]
        return self


    def forward_step(self, *args: _Tuple[torch.Tensor], **kwargs: _Mapping[str, _Any]) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            *args (*torch.Tensor): 输入
            **kwargs (str: Any): 输入
        Returns:
            res (torch.Tensor): 输出
        """
        raise NotImplementedError("Single step forward function for %s should be defined." % self.__class__.__name__)


    def forward_steps(self, *args: _Tuple[torch.Tensor], **kwargs: _Mapping[str, _Any]) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            *args (*torch.Tensor): 输入
            **kwargs (str: Any): 输入
        Returns:
            res (torch.Tensor): 输出
        """
        if not isinstance(args, _Tuple):
            args = (args,)
        time_steps = args[0].shape[0]
        def _forward_per_time_step(t, args_t, kwargs_t):
            _ = [module.single_step_mode_() if isinstance(module, Module) else None for module in self.children()] if not t else None
            return self.forward_step(*args_t, **kwargs_t)
        result = [_forward_per_time_step(t, tuple(x[t] if isinstance(x, torch.Tensor) else x for x in args), kwargs) for t in range(time_steps)]
        if isinstance(result[0], _Tuple):
            y = (torch.stack([r[col] for r in result]) if isinstance(el, torch.Tensor) else el for col, el in enumerate(result[0]))
        else:
            y = torch.stack(result)
        return y


    def forward(self, *args: _Tuple[torch.Tensor], **kwargs: _Mapping[str, _Any]) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            *args (*torch.Tensor): 输入
            **kwargs (str: Any): 输入
        Returns:
            res (torch.Tensor): 输出
        """
        if self.multi_step_mode:
            res = self.forward_steps(*args, **kwargs)
        else:
            res = self.forward_step(*args, **kwargs)
        return res