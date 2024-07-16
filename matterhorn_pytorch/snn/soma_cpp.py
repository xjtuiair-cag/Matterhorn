# -*- coding: UTF-8 -*-
"""
脉冲神经网络神经元的胞体，一层的后半段。输入为模拟电位值，输出为脉冲。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
此模块为其C++实现方案。
"""


import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from typing import Callable as _Callable, Any as _Any
from matterhorn_pytorch.snn.firing import Firing as _Firing, Gaussian as _Gaussian
import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.soma import LIF as _LIF
from matterhorn_pytorch.snn.soma_cpp_source import __fp_lif_source, __bp_lif_source


class multi_step_mode_lif(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor, u_init: torch.Tensor, tau_m: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, fp_lif: _Callable, bp_lif: _Callable) -> torch.Tensor:
        """
        多时间步LIF神经元前向传播的C++实现。
        Args:
            ctx (Any): 上下文
            x (torch.Tensor): 来自突触的输入电位$X_{i}^{l}(t)$
            u_init (torch.Tensor): 初始电位
            tau_m (torch.Tensor): 膜时间常数$τ_{m}$
            u_threshold (torch.Tensor): 阈电位$u_{th}$
            u_rest (torch.Tensor): 静息电位$u_{rest}$
            firing_mode (int): 发射脉冲的模式
            a (float): 参数$a$
            reset_mode (int): 重置的模式，有硬重置（0）和软重置（1）两种
        Returns:
            y (torch.Tensor): 输出
        """
        device = x.device
        time_steps = x.shape[0]
        if device.type != "cpu":
            x = x.to(device = torch.device("cpu"))
            u_init = u_init.to(device = torch.device("cpu"))
            tau_m = tau_m.to(device = torch.device("cpu"))
        o = torch.zeros_like(x)
        u = torch.zeros_like(x)
        h = torch.zeros_like(x)
        ctx.save_for_backward(x, u_init, tau_m, u_threshold, u_rest)
        ctx.fp_lif = fp_lif
        ctx.bp_lif = bp_lif
        ctx.fp_lif(time_steps, o, u, h, x, u_init, tau_m, u_rest, u_threshold)
        u_last = u[-1]
        if device.type != "cpu":
            o = o.to(device = device)
            u_last = u_last.to(device = device)
        return o, u_last
    

    @staticmethod
    def backward(ctx: _Any, grad_o: torch.Tensor, grad_u_last: torch.Tensor) -> torch.Tensor:
        """
        多时间步LIF神经元反向传播的C++实现。
        Args:
            ctx (Any): 上下文
            grad_o (torch.Tensor): 输出梯度
        Returns:
            grad_x (torch.Tensor): 输入梯度
            grad_u_init (torch.Tensor): 初始电位的梯度
            grad_tau_m (torch.Tensor): 膜时间常数$τ_{m}$的梯度
        """
        grad_o = grad_o.clone()
        device = grad_o.device
        x, u_init, tau_m, u_threshold, u_rest = ctx.saved_tensors
        o = torch.zeros_like(x)
        u = torch.zeros_like(x)
        h = torch.zeros_like(x)
        ctx.fp_lif(time_steps, o, u, h, x, u_init, tau_m, u_rest, u_threshold)
        if device.type != "cpu":
            grad_o = grad_o.to(device = torch.device("cpu"))
            grad_u_last = grad_u_last.to(device = torch.device("cpu"))
        time_steps = grad_o.shape[0]
        grad_u = torch.zeros_like(u)
        grad_h = torch.zeros_like(h)
        grad_h[-1] = grad_u_last
        grad_x = torch.zeros_like(x)
        grad_u_init = torch.zeros_like(u_init)
        grad_tau_m = torch.zeros_like(u_init)
        ctx.bp_lif(time_steps, grad_o, grad_u, grad_h, grad_x, grad_u_init, grad_tau_m, o, u, h, x, u_init, tau_m, u_rest, u_threshold)
        grad_tau_m = torch.sum(grad_tau_m)
        if device.type != "cpu":
            grad_x = grad_x.to(device = device)
            grad_u_init = grad_u_init.to(device = device)
            grad_tau_m = grad_tau_m.to(device = device)        
        return grad_x, grad_u_init, grad_tau_m, None, None, None, None


class LIF(_LIF):
    def __init__(self, tau_m: float = 2.0, u_threshold: float = -0.055, u_rest: float = -0.07, spiking_function: _Firing = _Gaussian, hard_reset: bool = True, trainable: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        super().__init__(
            tau_m = tau_m,
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            trainable = trainable,
            device = device,
            dtype = dtype
        )
        fp_lif_name, fp_lif_source = __fp_lif_source(self.spiking_function, self.hard_reset)
        bp_lif_name, bp_lif_source = __bp_lif_source(self.spiking_function, self.hard_reset)
        self.functions = load_inline(
            name = "lif",
            cpp_sources = [fp_lif_source, bp_lif_source],
            functions = [fp_lif_name, bp_lif_name],
            verbose = True
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([super().extra_repr(), "ext=%s" % '"CPP"'])


    def forward_steps(self, x: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 来自突触的输入电位$X_{i}^{l}(t)$
        Returns:
            o (torch.Tensor): 胞体当前的输出脉冲$O_{i}^{l}(t)$
        """
        self.check_if_reset(self.u, x[0])
        self.u = _SF.to(self.u, x[0])
        o, self.u = multi_step_mode_lif.apply(x, self.u, self.tau_m, self.u_threshold, self.u_rest, self.functions.fp_lif, self.functions.bp_lif)
        return o