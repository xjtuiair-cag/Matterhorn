# -*- coding: UTF-8 -*-
"""
脉冲神经网络的容器，用来容纳时间和空间维度的脉冲神经网络集合。
建议先在空间维度上构建完整的脉冲神经网络结构，再在多个时间步之内进行模拟。
"""


import torch
import torch.nn as nn
import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.skeleton import Module as _Module
from typing import Iterable as _Iterable, Mapping as _Mapping, Tuple as _Tuple


def _multi_step_mode_(module: _Module, if_on: bool) -> _Module:
    if if_on and not module.multi_step_mode:
        if module.supports_multi_step_mode:
            module.multi_step_mode_()
        else:
            module = Temporal(module)
    if not if_on and module.multi_step_mode:
        if module.supports_single_step_mode:
            module.single_step_mode_()
        elif isinstance(module, Temporal):
            module = module.module
        else:
            raise ValueError("Unsupported step mode conversion on module %s" % (module.__class__.__name__,))
    return module


class Container(_Module):
    def __init__(self) -> None:
        super().__init__()


class Spatial(Container, nn.Sequential):
    def __init__(self, *args) -> None:
        """
        SNN的空间容器，用法同nn.Sequential，加入一些特殊的作用于SNN的函数。
        Args:
            *args ([nn.Module]): 按空间顺序传入的各个模块
        """
        multi_step_mode = False
        for module in args:
            assert isinstance(module, _Module), "Module %s is not an SNN module." % (module.__class__.__name__)
            multi_step_mode = multi_step_mode or module.multi_step_mode
        Container.__init__(self)
        nn.Sequential.__init__(self, *args)
        self.multi_step_mode_(multi_step_mode)


    def multi_step_mode_(self, if_on: bool = True) -> nn.Module:
        """
        调整模型至多时间步模式。
        Args
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
        """
        for idx in range(len(self)):
            module = self[idx]
            self[idx] = _multi_step_mode_(module, if_on)
        return self


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播函数，默认接受的张量形状为[T,B,...]（需要将时间维度通过permute等函数转到最外）
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            y (torch.Tensor): 输出张量
        """
        y = nn.Sequential.forward(self, *args, **kwargs)
        return y


class Temporal(Container):
    def __init__(self, module: _Module) -> None:
        """
        SNN的时间容器，在多个时间步之内执行脉冲神经网络。
        Args:
            module (nn.Module): 所用来执行的单步模型
        """
        assert isinstance(module, _Module) and not module.multi_step_mode, "Temporal container can only accept SNN module with single step mode."
        super().__init__()
        self.multi_step_mode_()
        self.module = module


    @property
    def supports_single_step_mode(self) -> bool:
        """
        是否支持单个时间步。
        Returns:
            if_support (bool): 是否支持单个时间步
        """
        return False


class Sequential(Container, nn.Sequential):
    def __init__(self, *args, multi_step_mode: bool = False) -> None:
        """
        对Sequential进行重写，涵盖ANN与SNN的网络。
        Args:
            args (*nn.Module): 按空间顺序传入的各个模块
            multi_step_mode (nn.Module): 是否支持多时间步模式。
        """
        Container.__init__(self)
        nn.Sequential.__init__(self, *args)
        self.multi_step_mode_(multi_step_mode)


    def multi_step_mode_(self, if_on: bool = True) -> nn.Module:
        """
        调整模型至多时间步模式。
        Args
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
        """
        for idx in range(len(self)):
            module = self[idx]
            is_snn_module = isinstance(module, _Module)
            if is_snn_module:
                self[idx] = _multi_step_mode_(module, if_on)
        return self


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播函数，默认接受的张量形状为[T,B,...]（需要将时间维度通过permute等函数转到最外）
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            y (torch.Tensor): 输出张量
        """
        y = nn.Sequential.forward(self, *args, **kwargs)
        return y


class ModuleList(Container, nn.ModuleList):
    def __init__(self, modules: _Iterable[nn.Module] = None) -> None:
        Container.__init__(self)
        nn.ModuleList.__init__(
            self,
            modules = modules
        )


class ModuleDict(Container, nn.ModuleDict):
    def __init__(self, modules: _Mapping[str, nn.Module] = None) -> None:
        Container.__init__(self)
        nn.ModuleDict.__init__(
            self,
            modules = modules
        )


class Agent(_Module):
    def __init__(self, nn_module: nn.Module, force_spike_output: bool = False) -> None:
        """
        ANN套壳，用于使ANN模块带有mth.snn.Module的方法
        Args:
            module (nn.Module): torch.nn模型
            force_spike_output (bool): 是否强制脉冲输出
        """
        super().__init__()
        assert not isinstance(nn_module, _Module), "Already an SNN module."
        self.nn_module: nn.Module = nn_module
        self.force_spike_output = force_spike_output


    def multi_step_mode_(self, if_on: bool = True) -> nn.Module:
        """
        调整模型至多时间步模式。
        Args
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
        """
        if self.nn_module is not None:
            for name, module in self.nn_module.named_children():
                is_snn_module = isinstance(module, _Module)
                if is_snn_module:
                    setattr(self, name, _multi_step_mode_(module, if_on))
        return self


    def reset(self) -> nn.Module:
        """
        重置模型。
        """
        super().reset()
        if self.nn_module is not None:
            for name, module in self.nn_module.named_children():
                is_snn_module = isinstance(module, _Module)
                if is_snn_module:
                    module.reset()
        return self


    def detach(self) -> nn.Module:
        """
        将模型中的某些变量从其计算图中分离。
        """
        super().detach()
        if self.nn_module is not None:
            for name, module in self.nn_module.named_children():
                is_snn_module = isinstance(module, _Module)
                if is_snn_module:
                    module.detach()
        return self


    def forward_step(self, *args, **kwargs) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            *args: 输入
            **kwargs: 输入
        Returns:
            res (torch.Tensor): 输出
        """
        if not isinstance(args, _Tuple):
            args = (args,)
        res = self.nn_module(*args, **kwargs)
        return res


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            *args: 输入
            **kwargs: 输入
        Returns:
            res (torch.Tensor): 输出
        """
        res = super().forward(*args, **kwargs)
        if self.force_spike_output:
            if isinstance(res, _Tuple):
                res = (_SF.val_to_spike(y) if isinstance(y, torch.Tensor) else y for y in res)
            else:
                res = _SF.val_to_spike(res)
        return res