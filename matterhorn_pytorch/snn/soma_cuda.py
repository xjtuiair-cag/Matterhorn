# -*- coding: UTF-8 -*-
"""
脉冲神经网络神经元的胞体，一层的后半段。输入为模拟电位值，输出为脉冲。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
此模块为其CUDA实现方案。
"""


import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from typing import Callable as _Callable, Any as _Any
from matterhorn_pytorch.snn.firing import Firing as _Firing, Gaussian as _Gaussian
import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.soma import LIF as _LIF
import matterhorn_pytorch._ext.cuda as _ext_cu


class multi_step_mode_lif_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor, u_init: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, tau_m: torch.Tensor, fp_lif: _Callable, bp_lif: _Callable) -> torch.Tensor:
        ctx.time_steps = x.shape[0]
        ctx.shape = torch.prod(torch.tensor(x.shape[1:])).item()
        ctx.fp_lif = fp_lif
        ctx.bp_lif = bp_lif
        o = torch.zeros_like(x)
        u = torch.zeros_like(x)
        h = torch.zeros_like(x)
        ctx.fp_lif(ctx.time_steps, ctx.shape, o, u, h, x, u_init, u_threshold, u_rest, tau_m)
        ctx.save_for_backward(o, u, h, x, u_init, u_threshold, u_rest, tau_m)
        u_last = u[-1]
        return o, u_last
    

    @staticmethod
    def backward(ctx: _Any, grad_o: torch.Tensor, grad_u_last: torch.Tensor) -> torch.Tensor:
        grad_o = grad_o.clone()
        o, u, h, x, u_init, u_threshold, u_rest, tau_m = ctx.saved_tensors
        grad_u = torch.zeros_like(u)
        grad_h = torch.zeros_like(h)
        grad_h[-1] = grad_u_last
        grad_x = torch.zeros_like(x)
        grad_u_init = torch.zeros_like(u_init)
        grad_u_threshold = torch.zeros_like(u_threshold)
        grad_u_rest = torch.zeros_like(u_rest)
        grad_tau_m = torch.zeros_like(u_init)
        ctx.bp_lif(ctx.time_steps, ctx.shape, grad_o, grad_u, grad_h, grad_x, grad_u_init, grad_tau_m, o, u, h, x, u_init, u_threshold, u_rest, tau_m)
        grad_tau_m = torch.sum(grad_tau_m)
        return grad_x, grad_u_init, grad_tau_m, None, None, None, None, None


class LIF(_LIF):
    def __init__(self, u_threshold: float = -0.055, u_rest: float = -0.07, tau_m: float = 2.0, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, trainable: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
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
        fp_name, fp_dec, fp_source =  _ext_cu.fp_lif_source(self.spiking_function, self.hard_reset)
        bp_name, bp_dec, bp_source =  _ext_cu.bp_lif_source(self.spiking_function, self.hard_reset)
        self.functions = load_inline(
            name =  _ext_cu.purify_name("lif_cu_%s_%s_%s" % (self.spiking_function.__class__.__name__, self.spiking_function.extra_repr(), "zero" if self.hard_reset else "sub")),
            cpp_sources = [fp_dec, bp_dec],
            cuda_sources = [fp_source, bp_source],
            functions = [fp_name, bp_name],
            verbose = True
        )


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
        o, self.u = multi_step_mode_lif_cuda.apply(x, self.u, self.u_threshold, self.u_rest, self.tau_m, self.functions.fp_lif_cuda, self.functions.bp_lif_cuda)
        return o