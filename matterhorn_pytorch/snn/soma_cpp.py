# -*- coding: UTF-8 -*-
"""
脉冲神经网络神经元的胞体，一层的后半段。输入为模拟电位值，输出为脉冲。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
此模块为其C++实现方案。
"""


import torch
import torch.nn as nn


import matterhorn_pytorch.snn.functional as SF
from matterhorn_pytorch.snn.skeleton import Module
from matterhorn_pytorch.snn.soma import LIF as _LIF
from matterhorn_pytorch.snn import surrogate
from matterhorn_pytorch.snn.surrogate import Gaussian
try:
    from matterhorn_cpp_extensions import fp_lif, bp_lif
except:
    raise NotImplementedError("Please install Matterhorn C++ Extensions.")


try:
    from rich import print
except:
    pass


class multi_time_step_lif(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.Any, x: torch.Tensor, u_init: torch.Tensor, tau_m: torch.Tensor, u_threshold: float, u_rest: float, spiking_mode: int, a: float, reset_mode: float) -> torch.Tensor:
        """
        多时间步LIF神经元前向传播的C++实现。
        Args:
            ctx: 上下文
            x (torch.Tensor): 来自突触的输入电位$X_{i}^{l}(t)$
            u_init (torch.Tensor): 初始电位
            tau_m (torch.Tensor): 膜时间常数$τ_{m}$
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            spiking_mode (int): 发射脉冲的模式
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
        fp_lif(o, u, h, x, time_steps, u_init, tau_m, u_rest, u_threshold, reset_mode)
        ctx.save_for_backward(o, u, h, x, u_init, tau_m)
        ctx.u_threshold = u_threshold
        ctx.u_rest = u_rest
        ctx.spiking_mode = spiking_mode
        ctx.a = a
        ctx.reset_mode = reset_mode
        u_last = u[-1]
        if device.type != "cpu":
            o = o.to(device = device)
            u_last = u_last.to(device = device)
        return o, u_last
    

    @staticmethod
    def backward(ctx: torch.Any, grad_o: torch.Tensor, grad_u_last: torch.Tensor) -> torch.Tensor:
        """
        多时间步LIF神经元反向传播的C++实现。
        Args:
            ctx: 上下文
            grad_o (torch.Tensor): 输出梯度
        Returns:
            grad_x (torch.Tensor): 输入梯度
            grad_u_init (torch.Tensor): 初始电位的梯度
            grad_tau_m (torch.Tensor): 膜时间常数$τ_{m}$的梯度
        """
        grad_o = grad_o.clone()
        device = grad_o.device
        o, u, h, x, u_init, tau_m = ctx.saved_tensors
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
        bp_lif(grad_o, grad_u, grad_h, grad_x, grad_u_init, grad_tau_m, time_steps, o, u, h, x, u_init, tau_m, ctx.u_rest, ctx.u_threshold, ctx.spiking_mode, ctx.a, ctx.reset_mode)
        grad_tau_m = torch.sum(grad_tau_m)
        if device.type != "cpu":
            grad_x = grad_x.to(device = device)
            grad_u_init = grad_u_init.to(device = device)
            grad_tau_m = grad_tau_m.to(device = device)        
        return grad_x, grad_u_init, grad_tau_m, None, None, None, None, None


class LIF(_LIF):
    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["tau_m=%g, u_threshold=%g, u_rest=%g" % (self.tau_m, self.u_threshold, self.u_rest), super().extra_repr(), "ext=%s" % ('"cpp"',)])


    def forward_multi_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 来自突触的输入电位$X_{i}^{l}(t)$
        Returns:
            o (torch.Tensor): 胞体当前的输出脉冲$O_{i}^{l}(t)$
        """
        self.u = SF.init_tensor(self.u, x[0])
        o, self.u = multi_time_step_lif.apply(x, self.u, self.tau_m, self.u_threshold, self.u_rest, self.spiking_function_prototype, self.spiking_function.a, self.reset_function_prototype)
        return o