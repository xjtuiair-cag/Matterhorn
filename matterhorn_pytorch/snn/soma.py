# -*- coding: UTF-8 -*-
"""
脉冲神经网络神经元的胞体，一层的后半段。输入为模拟电位值，输出为脉冲。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
"""


import torch
import torch.nn as nn
import torch.nn.functional as _F
import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.skeleton import Module as _Module
from matterhorn_pytorch.snn.firing import Firing as _Firing, Gaussian as _Gaussian
from typing import Any as _Any, Tuple as _Tuple, Mapping as _Mapping, Callable as _Callable, Optional as _Optional


class Soma(_Module):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, batch_first: bool = False, return_states: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        神经元胞体骨架。
        Args:
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            spiking_function (Firing): 计算脉冲时所使用的阶跃函数
            hard_reset (bool): 是否为硬重置
            batch_first (bool): 第一维为批(True)还是时间(False)
            return_states (bool): 是否返回状态变量
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__()
        self.u_threshold = nn.Parameter(torch.tensor(u_threshold, device = device, dtype = dtype), requires_grad = False)
        self.u_rest = nn.Parameter(torch.tensor(u_rest, device = device, dtype = dtype), requires_grad = False)
        self.spiking_function: _Firing = spiking_function
        self.hard_reset = hard_reset
        self.batch_first = batch_first
        self.return_states = return_states


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["u_threshold=%g" % self.u_threshold, "u_rest=%g" % self.u_rest, "hard_reset=%s" % (self.hard_reset,), "batch_first=%s" % (self.batch_first,), "return_states=%s" % (self.return_states,)])
    

    @property
    def firing_str(self) -> str:
        return self.spiking_function.__class__.__name__.lower()


class IF(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, batch_first: bool = False, return_states: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        Integrate-and-Fire(IF)神经元。
        无泄漏过程，一阶电位变换公式为：
        $$\frac{du}{dt}=RI$$
        Args:
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            spiking_function (Firing): 计算脉冲时所使用的阶跃函数
            hard_reset (bool): 是否为硬重置
            batch_first (bool): 第一维为批(True)还是时间(False)
            return_states (bool): 是否返回状态变量
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            batch_first = batch_first,
            return_states = return_states,
            device = device,
            dtype = dtype
        )


    def forward(self, x: torch.Tensor, h: _Optional[torch.Tensor] = None) -> _Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入电位$X_{i}^{l}$
            h (torch.Tensor): 初始历史电位$H_{i}^{l}(0)$
        Returns:
            u (torch.Tensor): 输出脉冲$O_{i}^{l}(t)$
            h (torch.Tensor): 最终历史电位$H_{i}^{l}(T)$
        """
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if h is None:
            h = torch.full_like(x[0], self.u_rest)
        x, h = _SF.if_neuron(x, h, self.u_threshold, self.u_rest, self.firing_str, self.hard_reset)
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if self.return_states:
            return x, h
        return x


class LIF(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, tau_m: float = 2.0, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, batch_first: bool = False, return_states: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        Leaky-Integrate-and-Fire(LIF)神经元。
        一阶电位变换公式为：
        $$τ\frac{du}{dt}=-(u-u_{rest})+RI$$
        Args:
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            tau_m (float): 膜时间常数$τ_{m}$
            spiking_function (Firing): 计算脉冲时所使用的阶跃函数
            hard_reset (bool): 是否为硬重置
            batch_first (bool): 第一维为批(True)还是时间(False)
            return_states (bool): 是否返回状态变量
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            batch_first = batch_first,
            return_states = return_states,
            device = device,
            dtype = dtype
        )
        self.tau_m = nn.Parameter(torch.tensor(tau_m, device = device, dtype = dtype), requires_grad = False)


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([", ".join(["tau_m=%g" % self.tau_m]), Soma.extra_repr(self)])


    def forward(self, x: torch.Tensor, h: _Optional[torch.Tensor] = None) -> _Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入电位$X_{i}^{l}$
            h (torch.Tensor): 初始历史电位$H_{i}^{l}(0)$
        Returns:
            u (torch.Tensor): 输出脉冲$O_{i}^{l}(t)$
            h (torch.Tensor): 最终历史电位$H_{i}^{l}(T)$
        """
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if h is None:
            h = torch.full_like(x[0], self.u_rest)
        x, h = _SF.lif_neuron(x, h, self.u_threshold, self.u_rest, self.tau_m, self.firing_str, self.hard_reset)
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if self.return_states:
            return x, h
        return x


class QIF(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, tau_m: float = 2.0, u_c: float = 1.0, a_0: float = 1.0, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, batch_first: bool = False, return_states: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        Quadratic Integrate-and-Fire(QIF)神经元。
        一阶电位变换公式为：
        $$τ\frac{du}{dt}=a_{0}(u-u_{rest})(u-u_{c})+RI$$
        Args:
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            tau_m (float): 膜时间常数$τ_{m}$
            u_c (float): 参数$u_{c}$
            a_0 (float): 参数$a_{0}$
            spiking_function (Firing): 计算脉冲时所使用的阶跃函数
            hard_reset (bool): 是否为硬重置
            batch_first (bool): 第一维为批(True)还是时间(False)
            return_states (bool): 是否返回状态变量
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            batch_first = batch_first,
            return_states = return_states,
            device = device,
            dtype = dtype
        )
        self.tau_m = nn.Parameter(torch.tensor(tau_m, device = device, dtype = dtype), requires_grad = False)
        self.a_0 = nn.Parameter(torch.tensor(a_0, device = device, dtype = dtype), requires_grad = False)
        self.u_c = nn.Parameter(torch.tensor(u_c, device = device, dtype = dtype), requires_grad = False)
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([", ".join(["tau_m=%g" % self.tau_m, "a_0=%g" % self.a_0, "u_C=%g" % self.u_c]), Soma.extra_repr(self)])


    def forward(self, x: torch.Tensor, h: _Optional[torch.Tensor] = None) -> _Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入电位$X_{i}^{l}$
            h (torch.Tensor): 初始历史电位$H_{i}^{l}(0)$
        Returns:
            u (torch.Tensor): 输出脉冲$O_{i}^{l}(t)$
            h (torch.Tensor): 最终历史电位$H_{i}^{l}(T)$
        """
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if h is None:
            h = torch.full_like(x[0], self.u_rest)
        x, h = _SF.qif_neuron(x, h, self.u_threshold, self.u_rest, self.tau_m, self.u_c, self.a_0, self.firing_str, self.hard_reset)
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if self.return_states:
            return x, h
        return x


class ExpIF(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, tau_m: float = 2.0, u_t: float = 0.0, delta_t: float = 0.001, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, batch_first: bool = False, return_states: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        Exponential Integrate-and-Fire(ExpIF)神经元。
        一阶电位变换公式为：
        $$τ\frac{du}{dt}=-(u-u_{rest})+Δ_{T}e^{\frac{u-u_{T}}{Δ_{T}}}+RI$$
        Args:
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            tau_m (float): 膜时间常数$τ_{m}$
            u_t (float): 参数$u_{T}$
            delta_t (float): 参数$Δ_{T}$
            spiking_function (Firing): 计算脉冲时所使用的阶跃函数
            hard_reset (bool): 是否为硬重置
            batch_first (bool): 第一维为批(True)还是时间(False)
            return_states (bool): 是否返回状态变量
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            batch_first = batch_first,
            return_states = return_states,
            device = device,
            dtype = dtype
        )
        self.tau_m = nn.Parameter(torch.tensor(tau_m, device = device, dtype = dtype), requires_grad = False)
        self.delta_t = nn.Parameter(torch.tensor(delta_t, device = device, dtype = dtype), requires_grad = False)
        self.u_t = nn.Parameter(torch.tensor(u_t, device = device, dtype = dtype), requires_grad = False)
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([", ".join(["tau_m=%g" % self.tau_m, "u_t=%g" % self.u_t, "delta_t=%g" % self.delta_t]), Soma.extra_repr(self)])


    def forward(self, x: torch.Tensor, h: _Optional[torch.Tensor] = None) -> _Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入电位$X_{i}^{l}$
            h (torch.Tensor): 初始历史电位$H_{i}^{l}(0)$
        Returns:
            u (torch.Tensor): 输出脉冲$O_{i}^{l}(t)$
            h (torch.Tensor): 最终历史电位$H_{i}^{l}(T)$
        """
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if h is None:
            h = torch.full_like(x[0], self.u_rest)
        x, h = _SF.expif_neuron(x, h, self.u_threshold, self.u_rest, self.tau_m, self.u_t, self.delta_t, self.firing_str, self.hard_reset)
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if self.return_states:
            return x, h
        return x


class Izhikevich(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, a: float = 1.0, b: float = 1.0, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, batch_first: bool = False, return_states: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """, 
        Izhikevich神经元。
        一阶电位变换公式为：
        $$\frac{du}{dt}=0.04u^{2}+5u+140-w+I$$
        $$\frac{dw}{dt}=a(bu-w)$$
        Args:
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            u_t (float): 参数$u_{T}$
            delta_t (float): 参数$Δ_{T}$
            spiking_function (Firing): 计算脉冲时所使用的阶跃函数
            hard_reset (bool): 是否为硬重置
            batch_first (bool): 第一维为批(True)还是时间(False)
            return_states (bool): 是否返回状态变量
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            batch_first = batch_first,
            return_states = return_states,
            device = device,
            dtype = dtype
        )
        self.a = nn.Parameter(torch.tensor(a, device = device, dtype = dtype), requires_grad = False)
        self.b = nn.Parameter(torch.tensor(b, device = device, dtype = dtype), requires_grad = False)
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([", ".join(["a=%g" % self.a, "b=%g" % self.b]), Soma.extra_repr(self)])


    def forward(self, x: torch.Tensor, h_w: _Optional[_Tuple[torch.Tensor, torch.Tensor]] = None) -> _Tuple[torch.Tensor, _Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入电位$X_{i}^{l}$
            h_w (torch.Tensor): 初始历史电位$H_{i}^{l}(0)$与参数$W_{i}^{l}(0)$
        Returns:
            u (torch.Tensor): 输出脉冲$O_{i}^{l}(t)$
            h_w (torch.Tensor): 最终历史电位$H_{i}^{l}(T)$与参数$W_{i}^{l}(T)$
        """
        if h_w is None:
            h_w = (None, None)
        h, w = h_w
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if h is None:
            h = torch.full_like(x[0], self.u_rest)
        if w is None:
            w = torch.full_like(x[0], 0.0)
        x, (h, w) = _SF.izhikevich_neuron(x, (h, w), self.u_threshold, self.u_rest, self.a, self.b, self.firing_str, self.hard_reset)
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if self.return_states:
            return x, (h, w)
        return x


class KLIF(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, tau_m: float = 2.0, k: float = 2.0, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, batch_first: bool = False, return_states: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        KLIF神经元
        Args:
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            tau_m (float): 膜时间常数$τ_{m}$
            k (float): 参数k
            spiking_function (Firing): 计算脉冲时所使用的阶跃函数
            hard_reset (bool): 是否为硬重置
            batch_first (bool): 第一维为批(True)还是时间(False)
            return_states (bool): 是否返回状态变量
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            batch_first = batch_first,
            return_states = return_states,
            device = device,
            dtype = dtype
        )
        self.tau_m = nn.Parameter(torch.tensor(tau_m, device = device, dtype = dtype), requires_grad = False)
        self.k = nn.Parameter(torch.tensor(k, device = device, dtype = dtype), requires_grad = False)


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([", ".join(["tau_m=%g" % self.tau_m, "k=%g" % self.k]), Soma.extra_repr(self)])


    def forward(self, x: torch.Tensor, h: _Optional[torch.Tensor] = None) -> _Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入电位$X_{i}^{l}$
            h (torch.Tensor): 初始历史电位$H_{i}^{l}(0)$
        Returns:
            u (torch.Tensor): 输出脉冲$O_{i}^{l}(t)$
            h (torch.Tensor): 最终历史电位$H_{i}^{l}(T)$
        """
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if h is None:
            h = torch.full_like(x[0], self.u_rest)
        x, h = _SF.klif_neuron(x, h, self.u_threshold, self.u_rest, self.tau_m, self.k, self.firing_str, self.hard_reset)
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if self.return_states:
            return x, h
        return x


class LIAF(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, tau_m: float = 2.0, spiking_function: _Firing = _Gaussian(), activation_function: nn.Module = nn.ReLU(), hard_reset: bool = True, batch_first: bool = False, return_states: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        Leaky Integrate-and-Analog-Fire(LIAF)神经元
        Args:
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            tau_m (float): 膜时间常数$τ_{m}$
            spiking_function (Firing): 计算脉冲时所使用的阶跃函数
            activation_function (nn.Module): 激活函数
            hard_reset (bool): 是否为硬重置
            batch_first (bool): 第一维为批(True)还是时间(False)
            return_states (bool): 是否返回状态变量
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            batch_first = batch_first,
            return_states = return_states,
            device = device,
            dtype = dtype
        )
        self.tau_m = nn.Parameter(torch.tensor(tau_m, device = device, dtype = dtype), requires_grad = False)
        self.activation_function = activation_function


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([", ".join(["tau_m=%g" % self.tau_m]), Soma.extra_repr(self)])


    def forward(self, x: torch.Tensor, h: _Optional[torch.Tensor] = None) -> _Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入电位$X_{i}^{l}$
            h (torch.Tensor): 初始历史电位$H_{i}^{l}(0)$
        Returns:
            u (torch.Tensor): 输出脉冲$O_{i}^{l}(t)$
            h (torch.Tensor): 最终历史电位$H_{i}^{l}(T)$
        """
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if h is None:
            h = torch.full_like(x[0], self.u_rest)
        u, h = _SF.lim_neuron(x, h, self.u_threshold, self.u_rest, self.tau_m, self.firing_str, self.hard_reset)
        x = self.activation_function(u)
        if self.batch_first:
            x = x.swapaxes(0, 1)
        if self.return_states:
            return x, h
        return x