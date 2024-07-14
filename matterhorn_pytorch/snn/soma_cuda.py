# -*- coding: UTF-8 -*-
"""
脉冲神经网络神经元的胞体，一层的后半段。输入为模拟电位值，输出为脉冲。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
此模块为其CUDA实现方案。
"""


import torch
import torch.nn as nn


import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.soma import LIF as _LIF
from typing import Any as _Any
try:
    from matterhorn_cuda_extensions import cu_fp_lif, cu_bp_lif
except:
    raise NotImplementedError("Please install Matterhorn CUDA Extensions.")


import numpy as np


class multi_step_mode_lif_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor, u_init: torch.Tensor, tau_m: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, firing_mode: int, a: float, reset_mode: float) -> torch.Tensor:
        """
        多时间步LIF神经元前向传播的C++实现。
        Args:
            ctx (Any): 上下文
            x (torch.Tensor): 来自突触的输入电位$X_{i}^{l}(t)$
            u_init (torch.Tensor): 初始电位
            tau_m (torch.Tensor): 膜时间常数$τ_{m}$
            u_threshold (at::Tensor): 阈电位$u_{th}$
            u_rest (torch.Tensor): 静息电位$u_{rest}$
            firing_mode (int): 发射脉冲的模式
            a (float): 参数$a$
            reset_mode (int): 重置的模式，有硬重置（0）和软重置（1）两种
        Returns:
            y (torch.Tensor): 输出
        """
        x = x.clone()
        u_init = u_init.clone()
        tau_m = tau_m.clone()
        device = x.device
        assert device.type == "cuda", "You must use CUDA tensors."
        time_steps = x.shape[0]
        shape = np.prod(x.shape[1:])
        o = torch.zeros_like(x)
        u = torch.zeros_like(x)
        h = torch.zeros_like(x)
        cu_fp_lif(o, u, h, x, time_steps, shape, u_init, tau_m, u_rest, u_threshold, firing_mode, reset_mode)
        ctx.save_for_backward(x, u_init, tau_m, u_threshold, u_rest)
        ctx.firing_mode = firing_mode
        ctx.a = a
        ctx.reset_mode = reset_mode
        u_last = u[-1]
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
        """
        grad_o = grad_o.clone()
        grad_u_last = grad_u_last.clone()
        device = grad_o.device
        assert device.type == "cuda", "You must use CUDA tensors."
        x, u_init, tau_m, u_threshold, u_rest = ctx.saved_tensors
        o = torch.zeros_like(x)
        u = torch.zeros_like(x)
        h = torch.zeros_like(x)
        cu_fp_lif(o, u, h, x, time_steps, shape, u_init, tau_m, u_rest, u_threshold, ctx.firing_mode, ctx.reset_mode)
        time_steps = grad_o.shape[0]
        shape = np.prod(grad_o.shape[1:])
        grad_u = torch.zeros_like(u)
        grad_h = torch.zeros_like(h)
        grad_h[-1] = grad_u_last
        grad_x = torch.zeros_like(x)
        grad_u_init = torch.zeros_like(u_init)
        grad_tau_m = torch.zeros_like(u_init)
        cu_bp_lif(grad_o, grad_u, grad_h, grad_x, grad_u_init, grad_tau_m, time_steps, shape, o, u, h, x, u_init, tau_m, u_rest, u_threshold, ctx.firing_mode, ctx.a, ctx.reset_mode)
        grad_tau_m = torch.sum(grad_tau_m)
        return grad_x, grad_u_init, grad_tau_m, None, None, None, None, None


class LIF(_LIF):
    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([super().extra_repr(), "ext=%s" % '"CUDA"'])


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
        supported_spiking_functions = ("Rectangular", "Polynomial", "Sigmoid", "Gaussian")
        current_spiking_function = self.spiking_function.__class__.__name__
        assert current_spiking_function in supported_spiking_functions, "Unsupported spiking function"
        spiking_function_prototype = supported_spiking_functions.index(current_spiking_function)
        a = getattr(self.spiking_function, "a") if hasattr(self.spiking_function, "a") else 1.0
        reset_function_prototype = 0 if self.hard_reset else 1
        o, self.u = multi_step_mode_lif_cuda.apply(x, self.u, self.tau_m, self.u_threshold, self.u_rest, spiking_function_prototype, a, reset_function_prototype)
        return o