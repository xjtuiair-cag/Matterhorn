# -*- coding: UTF-8 -*-
"""
SNN模块的框架，在torch.nn的基础上，定义了几个SNN的基本函数。
"""


import torch
import torch.nn as nn
from typing import Union
import matterhorn_pytorch.snn.functional as SF
try:
    from rich import print
except:
    pass


def reset_hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
    if isinstance(model, Module) and model.multi_time_step and model.reset_after_process:
        model.reset()


class Module(nn.Module):
    def __init__(self, multi_time_step: bool = False, reset_after_process: bool = False) -> None:
        """
        脉冲神经网络模块的骨架。
        Args:
            multi_time_step (bool): 是否调整为多个时间步模式
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
        """
        nn.Module.__init__(self)
        self._multi_time_step = False
        self._reset_after_process = reset_after_process
        if multi_time_step:
            self.multi_time_step_(multi_time_step)
        self.register_forward_hook(reset_hook)


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return super().extra_repr()


    def supports_single_time_step(self) -> bool:
        """
        是否支持单个时间步。
        Returns:
            if_support (bool): 是否支持单个时间步
        """
        return True


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
        if self.supports_multi_time_step() and if_on:
            self._multi_time_step = True
        elif self.supports_single_time_step() and not if_on:
            self._multi_time_step = False
        for module in self.children():
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.multi_time_step_(if_on)
                assert module.multi_time_step == self.multi_time_step, "Unmatched step mode"
        return self


    @property
    def reset_after_process(self) -> bool:
        """
        是否在执行完后自动重置。
        Returns:
            if_on (bool): 是否为自动重置（True为自动重置，False为手动重置）
        """
        return self._reset_after_process


    def reset_after_process_(self, if_on: bool) -> nn.Module:
        """
        调整是否在执行完后自动重置。
        Args:
            if_on (bool): 是否为自动重置（True为自动重置，False为手动重置）
        """
        self._reset_after_process = if_on
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


class Agent(Module):
    def __init__(self, nn_module: nn.Module, force_spike_output: bool = False, multi_time_step: bool = False, reset_after_process: bool = False) -> None:
        """
        ANN套壳，用于使ANN模块带有mth.snn.Module的方法
        Args:
            module (nn.Module): torch.nn模型
            force_spike_output (bool): 是否强制脉冲输出
            multi_time_step (bool): 是否调整为多个时间步模式
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
        """
        is_snn_module = isinstance(nn_module, Module)
        assert not is_snn_module, "Already an SNN module."
        super().__init__(
            multi_time_step = multi_time_step,
            reset_after_process = reset_after_process
        )
        self.nn_module: nn.Module = nn_module
        self.force_spike_output = force_spike_output


    def multi_time_step_(self, if_on: bool) -> nn.Module:
        """
        调整模型的多时间步模式。
        Args
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
        """
        if self.supports_multi_time_step() and if_on:
            self._multi_time_step = True
        elif self.supports_single_time_step() and not if_on:
            self._multi_time_step = False
        for module in self.nn_module.children():
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.multi_time_step_(if_on)
                assert module.multi_time_step == self.multi_time_step, "Unmatched step mode"
        return self


    def reset(self) -> nn.Module:
        """
        重置模型。
        """
        for module in self.nn_module.children():
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.reset()
        return self


    def detach(self) -> nn.Module:
        """
        将模型中的某些变量从其计算图中分离。
        """
        for module in self.nn_module.children():
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.detach()
        return self


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 输入张量，形状为[B, ...]
        Returns:
            y (torch.Tensor): 输出张量，形状为[B, ...]
        """
        y = self.nn_module(x)
        return y


    def forward_multi_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 输入张量，形状为[T, B, ...]
        Returns:
            y (torch.Tensor): 输出张量，形状为[T, B, ...]
        """
        time_steps = x.shape[0]
        batch_size = x.shape[1]
        x = x.flatten(0, 1)
        y = self.forward_single_time_step(x)
        output_shape = [time_steps, batch_size] + list(y.shape[1:])
        y = y.reshape(output_shape)
        return y


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            y (torch.Tensor): 输出张量
        """
        if self.multi_time_step:
            y = self.forward_multi_time_step(x)
        else:
            y = self.forward_single_time_step(x)
        if self.force_spike_output:
            y = SF.val_to_spike(y)
        return y