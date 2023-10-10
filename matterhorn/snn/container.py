# -*- coding: UTF-8 -*-
"""
脉冲神经网络的容器，用来容纳时间和空间维度的脉冲神经网络集合。
建议先在空间维度上构建完整的脉冲神经网络结构，再在多个时间步之内进行模拟。
"""


import torch
import torch.nn as nn
from matterhorn.snn.skeleton import Module
from matterhorn.snn.encoder import Encoder
from matterhorn.snn.decoder import Decoder
from typing import Optional
try:
    from rich import print
except:
    pass


class Container(Module):
    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return ""


class Spatial(Container, nn.Sequential):
    def __init__(self, *args) -> None:
        """
        SNN的空间容器，用法同nn.Sequential，加入一些特殊的作用于SNN的函数。
        @params:
            *args: [nn.Module] 按空间顺序传入的各个模块
        """
        Container.__init__(self)
        nn.Sequential.__init__(self, *args)
        self.multi_time_step__ = False
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                self.multi_time_step__ = self.multi_time_step__ or module.multi_time_step


    def supports_single_time_step(self) -> bool:
        """
        是否支持单个时间步。
        @return:
            if_support: bool 是否支持单个时间步
        """
        return True


    def supports_multi_time_step(self) -> bool:
        """
        是否支持多个时间步。
        @return:
            if_support: bool 是否支持多个时间步
        """
        return True


    def multi_time_step_(self, if_on: bool) -> bool:
        """
        调整模型的多时间步模式。
        @params
            if_on: bool 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
        """
        self.multi_time_step__ = False
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                res = module.multi_time_step_()
                self.multi_time_step__ = self.multi_time_step__ or module.multi_time_step
        return True


    def reset(self) -> None:
        """
        一次重置该序列中所有的神经元。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.reset()


    def start_step(self) -> None:
        """
        开始STDP训练。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.start_step()


    def stop_step(self) -> None:
        """
        停止STDP训练。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.stop_step()
    

    def step_once(self) -> None:
        """
        一次部署所有结点的STDP训练。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.step_once()


class Temporal(Container):
    def __init__(self, module: nn.Module, reset_after_process = True) -> None:
        """
        SNN的时间容器，在多个时间步之内执行脉冲神经网络。
        @params:
            module: nn.Module 所用来执行的单步模型
            reset_after_process: bool 是否在执行完后自动重置，若为False则需要手动重置
        """
        is_snn_module = isinstance(module, Module)
        if is_snn_module:
            assert module.multi_time_step == False, "You cannot put a multi time step module %s into temporal container" % (module.__class__.__name__,)
        super().__init__(
            multi_time_step = True
        )
        self.module = module
        self.reset_after_process = reset_after_process
        self.step_after_process = False


    def supports_single_time_step(self) -> bool:
        """
        是否支持单个时间步。
        @return:
            if_support: bool 是否支持单个时间步
        """
        return False


    def supports_multi_time_step(self) -> bool:
        """
        是否支持多个时间步。
        @return:
            if_support: bool 是否支持多个时间步
        """
        return True


    def reset(self) -> None:
        """
        重置模型。
        """
        is_snn_module = isinstance(self.module, Module)
        if is_snn_module:
            self.module.reset()

    
    def start_step(self) -> None:
        """
        开始STDP训练。
        """
        self.step_after_process = True
        is_snn_module = isinstance(self.module, Module)
        if is_snn_module:
            self.module.start_step()
    

    def stop_step(self) -> None:
        """
        停止STDP训练。
        """
        self.step_after_process = False
        is_snn_module = isinstance(self.module, Module)
        if is_snn_module:
            self.module.stop_step()
    

    def step_once(self) -> None:
        """
        部署结点的STDP训练。
        """
        is_snn_module = isinstance(self.module, Module)
        if is_snn_module:
            self.module.step_once()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，默认接受的张量形状为[T,B,...]（需要将时间维度通过permute等函数转到最外）
        @params:
            x: torch.Tensor 输入张量
        @return:
            y: torch.Tensor 输出张量
        """
        time_steps = x.shape[0]
        result = []
        for t in range(time_steps):
            result.append(self.module(x[t]))
        y = torch.stack(result)
        if self.step_after_process:
            self.step_once()
        if self.reset_after_process:
            self.reset()
        return y


class Sequential(Container, nn.Sequential):
    def __init__(self, *args, reset_after_process = True) -> None:
        """
        对Sequential进行重写，涵盖ANN与SNN的网络。
        @params:
            *args: [nn.Module] 按空间顺序传入的各个模块
        """
        Container.__init__(self, multi_time_step = True)
        nn.Sequential.__init__(self, *args)
        convert_indices = []
        remove_indices = []
        last_single_step_idx = -2
        last_snn_idx = -2
        for module_idx in range(len(self)):
            module = self[module_idx]
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                if module_idx and last_snn_idx != module_idx - 1:
                    assert isinstance(module, Encoder), "For ANNs(%s) -> SNNs(%s) you must add an encoder." % (self[module_idx - 1].__class__.__name__, module.__class__.__name__)
                is_multi_time_step = module.multi_time_step
                if not is_multi_time_step:
                    if module.supports_multi_time_step():
                        module.multi_time_step_(True)
                        self[module_idx] = module
                    else:
                        if last_single_step_idx == module_idx - 1:
                            convert_indices[-1].append(module_idx)
                            remove_indices.append(module_idx)
                        else:
                            convert_indices.append([module_idx, module_idx])
                        last_single_step_idx = module_idx
                else:
                    pass
                last_snn_idx = module_idx
            else:
                if last_snn_idx == module_idx - 1:
                    assert isinstance(self[last_snn_idx], Decoder), "For SNNs(%s) -> ANNs(%s) you must add an encoder." % (self[module_idx - 1].__class__.__name__, module.__class__.__name__)
        for indices in convert_indices:
            if len(indices) == 2:
                self[indices[0]] = Temporal(
                    self[indices[1]],
                    reset_after_process = reset_after_process
                )
            else:
                module_list = []
                for module_idx in indices[1:]:
                    module_list.append(self[module_idx])
                self[indices[0]] = Temporal(
                    Spatial(
                        *module_list
                    ),
                    reset_after_process = reset_after_process
                )
        remove_indices = remove_indices[::-1]
        for idx in remove_indices:
            del self[idx]


    def supports_single_time_step(self) -> bool:
        """
        是否支持单个时间步。
        @return:
            if_support: bool 是否支持单个时间步
        """
        return False


    def supports_multi_time_step(self) -> bool:
        """
        是否支持多个时间步。
        @return:
            if_support: bool 是否支持多个时间步
        """
        return True


    def reset(self) -> None:
        """
        一次重置该序列中所有的神经元。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.reset()


    def start_step(self) -> None:
        """
        开始STDP训练。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.start_step()


    def stop_step(self) -> None:
        """
        停止STDP训练。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.stop_step()
    

    def step_once(self) -> None:
        """
        一次部署所有结点的STDP训练。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.step_once()