# -*- coding: UTF-8 -*-
"""
SNN模块的框架，在torch.nn的基础上，定义了几个SNN的基本函数。
"""


import warnings
import torch
import torch.nn as nn
from typing import Any as _Any, Tuple as _Tuple, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional


class Module(nn.Module):
    def __init__(self) -> None:
        """
        脉冲神经网络模块的骨架。
        """
        nn.Module.__init__(self)


    def _fold_for_parallel(self, x: torch.Tensor, target_dim: _Optional[int] = None) -> _Tuple[torch.Tensor, _Iterable[int]]:
        """
        将前几个维度压缩，以适应nn.Module中的预定义模块。
        Args:
            x (torch.Tensor): 未压缩前的张量
            target_dim (int): 目标维度
        Returns:
            x (torch.Tensor): 压缩后的张量
            shape (Tuple): 被压缩的形状信息
        """
        target_dim = target_dim if target_dim is not None else (x.ndim - 1)
        flatten_dims = x.ndim - target_dim
        shape = []
        if flatten_dims > 0:
            shape = list(x.shape[:flatten_dims + 1])
            x = x.flatten(0, flatten_dims)
        return x, shape


    def _unfold_from_parallel(self, x: torch.Tensor, shape: _Iterable[int]) -> torch.Tensor:
        """
        解压因并行而压缩的维度。
        Args:
            x (torch.Tensor): 压缩后的张量
            shape (Tuple): 被压缩的形状信息
        Returns:
            x (torch.Tensor): 未压缩前的张量
        """
        if len(shape):
            x = x.unflatten(0, shape)
        return x


    @property
    def multi_step_mode(self) -> bool:
        """
        当前是否为多时间步模式。
        Returns:
            if_on (bool): 当前是否为多个时间步模式
        """
        warnings.warn("Since Matterhorn 2.0.0, all modules will be multi-step, where the `multi_step_mode` attribute and `multi_step_mode_` function will be soon deprecated.", DeprecationWarning)
        return True
    

    @property
    def single_step_mode(self) -> bool:
        """
        当前是否为单时间步模式。
        Returns:
            if_on (bool): 当前是否为单个时间步模式
        """
        warnings.warn("Since Matterhorn 2.0.0, all modules will be multi-step, where the `multi_step_mode` attribute and `multi_step_mode_` function will be soon deprecated.", DeprecationWarning)
        return False


    def multi_step_mode_(self, if_on: bool = True, recursive: bool = True) -> nn.Module:
        """
        调整模型至多时间步模式。
        Args:
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
            recursive (bool): 是否递归调整子模块的时间步模式
        """
        warnings.warn("Since Matterhorn 2.0.0, all modules will be multi-step, where the `multi_step_mode` attribute and `multi_step_mode_` function will be soon deprecated.", DeprecationWarning)
        return self
    

    def single_step_mode_(self, if_on: bool = True, recursive: bool = True) -> nn.Module:
        """
        调整模型至单时间步模式。
        Args
            if_on (bool): 当前需要调整为什么模式（True为单时间步模式，False为多时间步模式）
        """
        return self


    def reset(self) -> nn.Module:
        """
        重置模型。
        """
        warnings.warn("Since Matterhorn 2.0.0, all modules will not have memory storage, where the `reset` function will be soon deprecated.", DeprecationWarning)
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
        warnings.warn("Since Matterhorn 2.0.0, all modules will be multi-step, where the `forward_step` and `forward_steps` function will be deprecated. Please define `forward` function to make the module work.", DeprecationWarning)
        if hasattr(self, "forward_steps"):
            res = self.forward_steps(*args, **kwargs)
        else:
            res = self.forward_step(*args, **kwargs)
        return res