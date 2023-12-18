# -*- coding: UTF-8 -*-
"""
SNN模块的框架，在torch.nn的基础上，定义了几个SNN的基本函数。
"""


import torch
import torch.nn as nn
from typing import Union
try:
    from rich import print
except:
    pass


class Module(nn.Module):
    def __init__(self, multi_time_step: bool = False, reset_after_process: bool = False) -> None:
        """
        脉冲神经网络模块的骨架。
        Args:
            multi_time_step (bool): 是否调整为多个时间步模式
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
        """
        nn.Module.__init__(self)
        self.multi_time_step__ = False
        self.reset_after_process__ = reset_after_process
        if multi_time_step:
            self.multi_time_step_(multi_time_step)
    

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
        return self.multi_time_step__


    def multi_time_step_(self, if_on: bool) -> nn.Module:
        """
        调整模型的多时间步模式。
        Args
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
        """
        if self.supports_multi_time_step() and if_on:
            self.multi_time_step__ = True
        elif self.supports_single_time_step() and not if_on:
            self.multi_time_step__ = False
        return self


    @property
    def reset_after_process(self) -> bool:
        """
        是否在执行完后自动重置。
        Returns:
            if_on (bool): 是否为自动重置（True为自动重置，False为手动重置）
        """
        return self.reset_after_process__


    def reset_after_process_(self, if_on: bool) -> bool:
        """
        调整是否在执行完后自动重置。
        Args:
            if_on (bool): 是否为自动重置（True为自动重置，False为手动重置）
        """
        self.reset_after_process__ = if_on


    def reset(self) -> None:
        """
        重置模型。
        """
        pass


    def detach(self) -> None:
        """
        将模型中的某些变量从其计算图中分离。
        """
        pass

    
    def train(self, mode: Union[str, bool] = "bp") -> None:
        """
        切换训练和测试模式。
        Args:
            mode (str | bool): 采用何种训练方式，None为测试模式
        """
        if mode is None:
            super().train(False)
            return
        if isinstance(mode, bool):
            super().train(mode)
            return
        if isinstance(mode, str):
            mode = mode.lower()
        if mode == "bp":
            super().train(True)
            return
        super().train(False)
    

    def eval(self) -> None:
        """
        切换测试模式。
        """
        super().eval()
    

    def step(self) -> None:
        """
        部署结点的自定义训练。
        """
        pass