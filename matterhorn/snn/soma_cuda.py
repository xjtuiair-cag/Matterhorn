import torch
import torch.nn as nn


from matterhorn.snn.skeleton import Module
from matterhorn.snn.soma import Soma
from matterhorn.snn import surrogate
try:
    from matterhorn_cuda_extensions import cu_fp_lif_heaviside_hard, cu_bp_lif_rectangular_hard
except:
    raise NotImplementedError("Please install Matterhorn CUDA Extensions.")


import numpy as np
from typing import Any
try:
    from rich import print
except:
    pass


class SomaCUDA(Soma):
    supported_surrogate_gradients = ("Rectangular",)

    def __init__(self, tau_m: float = 1.0, u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: Module = surrogate.Rectangular(), trainable: bool = False) -> None:
        """
        Response-Firing-Reset三段式神经元胞体骨架，分别为：
        （1）通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$；
        （2）通过当前电位$U_{i}^{l}(t)$计算当前脉冲$O_{i}^{l}(t)$；
        （3）通过当前脉冲$O_{i}^{l}(t)$重置当前电位$U_{i}^{l}(t)$。
        @params:
            tau_m: float 膜时间常数$τ_{m}$
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            spiking_function: nn.Module 计算脉冲时所使用的阶跃函数
            trainable: bool 参数是否可以训练
        """
        super().__init__(
            tau_m = tau_m,
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            trainable = trainable
        )
        surrogate_str = spiking_function.__class__.__name__
        assert surrogate_str in self.supported_surrogate_gradients, "Unknown surrogate gradient."
        self.spiking_function_prototype = self.supported_surrogate_gradients.index(surrogate_str)
        self.multi_time_step_(True)


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


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        @params:
            x: torch.Tensor 来自突触的输入电位$X_{i}^{l}(t)$
        @return:
            o: torch.Tensor 胞体当前的输出脉冲$O_{i}^{l}(t)$
        """
        self.u = self.init_tensor(self.u, x)
        self.u = self.f_response(self.u, x)
        o = self.f_firing(self.u)
        self.u = self.f_reset(self.u, o)
        return o


    def forward_multi_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        @params:
            x: torch.Tensor 来自突触的输入电位$X_{i}^{l}(t)$
        @return:
            o: torch.Tensor 胞体当前的输出脉冲$O_{i}^{l}(t)$
        """
        time_steps = x.shape[0]
        o_seq = []
        for t in time_steps:
            o_seq.append(self.forward_single_time_step(x[t]))
        o = torch.stack(o_seq)
        return o


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            x: torch.Tensor 来自突触的输入电位$X_{i}^{l}(t)$
        @return:
            o: torch.Tensor 胞体当前的输出脉冲$O_{i}^{l}(t)$
        """
        if self.multi_time_step:
            o = self.forward_multi_time_step(x)
        else:
            o = self.forward_single_time_step(x)
        return o


class multi_time_step_lif_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, u_init: torch.Tensor, tau_m: torch.Tensor, u_threshold: float, u_rest: float) -> torch.Tensor:
        """
        多时间步LIF神经元前向传播的C++实现。
        @params:
            ctx: 上下文
            x: torch.Tensor 来自突触的输入电位$X_{i}^{l}(t)$
            u_init: torch.Tensor 初始电位
            tau_m: torch.Tensor 膜时间常数$τ_{m}$
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
        @return:
            y: torch.Tensor 输出
        """
        device = x.device
        assert device.type == "cuda", "You must use CUDA tensors."
        time_steps = x.shape[0]
        shape = np.prod(x.shape[1:])
        o = torch.zeros_like(x)
        u = torch.zeros_like(x)
        h = torch.zeros_like(x)
        cu_fp_lif_heaviside_hard(o, u, h, x, time_steps, shape, u_init, tau_m, u_rest, u_threshold)
        ctx.save_for_backward(o, u, h, x, u_init, tau_m)
        ctx.u_threshold = u_threshold
        ctx.u_rest = u_rest
        if device.type != "cpu":
            o = o.to(device = device)
        return o
    

    @staticmethod
    def backward(ctx: Any, grad_o: torch.Tensor) -> torch.Tensor:
        """
        多时间步LIF神经元反向传播的C++实现。
        @params:
            ctx: 上下文
            grad_o: torch.Tensor 输出梯度
        @return:
            grad_x: torch.Tensor 输入梯度
        """
        device = grad_o.device
        assert device.type == "cuda", "You must use CUDA tensors."
        o, u, h, x, u_init, tau_m = ctx.saved_tensors
        time_steps = grad_o.shape[0]
        shape = np.prod(grad_o.shape[1:])
        grad_u = torch.zeros_like(u)
        grad_h = torch.zeros_like(h)
        grad_x = torch.zeros_like(x)
        grad_u_init = torch.zeros_like(u_init)
        grad_tau_m = torch.zeros_like(tau_m)
        cu_bp_lif_rectangular_hard(grad_o, grad_u, grad_h, grad_x, grad_u_init, grad_tau_m, time_steps, shape, o, u, h, x, u_init, tau_m, ctx.u_rest, ctx.u_threshold)
        return grad_x, grad_u_init, grad_tau_m, None, None


class LIF(SomaCUDA):
    def __init__(self, tau_m: float = 2.0, u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: nn.Module = surrogate.Rectangular(), trainable: bool = False) -> None:
        """
        Leaky-Integrate-and-Fire(LIF)神经元。
        一阶电位变换公式为：
        $$τ\frac{du}{dt}=-(u-u_{rest})+RI$$
        @params:
            tau_m: float 膜时间常数$τ_{m}$
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            spiking_function: nn.Module 计算脉冲时所使用的阶跃函数
            trainable: bool 参数是否可以训练
        """
        super().__init__(
            tau_m = tau_m,
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            trainable = trainable
        )
        self.multi_time_step_function = multi_time_step_lif_cuda()


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return super().extra_repr() + "tau_m=%.3f, u_th=%.3f, u_rest=%.3f" % (self.tau_m, self.u_threshold, self.u_rest)


    def forward_multi_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        @params:
            x: torch.Tensor 来自突触的输入电位$X_{i}^{l}(t)$
        @return:
            o: torch.Tensor 胞体当前的输出脉冲$O_{i}^{l}(t)$
        """
        self.u = self.init_tensor(self.u, x[0])
        o = self.multi_time_step_function.apply(x, self.u, self.tau_m, self.u_threshold, self.u_rest)
        return o