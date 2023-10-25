import torch
import torch.nn as nn


from matterhorn_pytorch.snn.skeleton import Module
from matterhorn_pytorch.snn.soma import Soma
from matterhorn_pytorch.snn import surrogate
try:
    from matterhorn_cpp_extensions import fp_lif, bp_lif
except:
    raise NotImplementedError("Please install Matterhorn C++ Extensions.")


from typing import Any
try:
    from rich import print
except:
    pass


class SomaCPP(Soma):
    supported_surrogate_gradients = ("Rectangular", "Polynomial", "Sigmoid", "Gaussian")

    def __init__(self, tau_m: float = 1.0, u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: Module = surrogate.Rectangular(), hard_reset: bool = True, trainable: bool = False) -> None:
        """
        Response-Firing-Reset三段式神经元胞体骨架，分别为：
        （1）通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$；
        （2）通过当前电位$U_{i}^{l}(t)$计算当前脉冲$O_{i}^{l}(t)$；
        （3）通过当前脉冲$O_{i}^{l}(t)$重置当前电位$U_{i}^{l}(t)$。
        @params:
            tau_m: float 膜时间常数$τ_{m}$
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            spiking_function: Module 计算脉冲时所使用的阶跃函数
            hard_reset: bool 是否为硬重置
            trainable: bool 参数是否可以训练
        """
        super().__init__(
            tau_m = tau_m,
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            trainable = trainable
        )
        surrogate_str = spiking_function.__class__.__name__
        assert surrogate_str in self.supported_surrogate_gradients, "Unknown surrogate gradient."
        self.spiking_function_prototype = self.supported_surrogate_gradients.index(surrogate_str)
        self.reset_prototype = 0 if self.hard_reset else 1
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


class multi_time_step_lif(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, u_init: torch.Tensor, tau_m: torch.Tensor, u_threshold: float, u_rest: float, spiking_mode: int, a: float, reset_mode: float) -> torch.Tensor:
        """
        多时间步LIF神经元前向传播的C++实现。
        @params:
            ctx: 上下文
            x: torch.Tensor 来自突触的输入电位$X_{i}^{l}(t)$
            u_init: torch.Tensor 初始电位
            tau_m: torch.Tensor 膜时间常数$τ_{m}$
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            spiking_mode: int 发射脉冲的模式
            a: float 参数$a$
            reset_mode: int 重置的模式，有硬重置（0）和软重置（1）两种
        @return:
            y: torch.Tensor 输出
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
    def backward(ctx: Any, grad_o: torch.Tensor, grad_u_last: torch.Tensor) -> torch.Tensor:
        """
        多时间步LIF神经元反向传播的C++实现。
        @params:
            ctx: 上下文
            grad_o: torch.Tensor 输出梯度
        @return:
            grad_x: torch.Tensor 输入梯度
            grad_u_init: torch.Tensor 初始电位的梯度
            grad_tau_m: torch.Tensor 膜时间常数$τ_{m}$的梯度
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
        grad_tau_m = torch.zeros_like(tau_m)
        bp_lif(grad_o, grad_u, grad_h, grad_x, grad_u_init, grad_tau_m, time_steps, o, u, h, x, u_init, tau_m, ctx.u_rest, ctx.u_threshold, ctx.spiking_mode, ctx.a, ctx.reset_mode)
        if device.type != "cpu":
            grad_x = grad_x.to(device = device)
            grad_u_init = grad_u_init.to(device = device)
            grad_tau_m = grad_tau_m.to(device = device)
        return grad_x, grad_u_init, grad_tau_m, None, None, None, None, None


class LIF(SomaCPP):
    def __init__(self, tau_m: float = 2.0, u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: Module = surrogate.Rectangular(), hard_reset: bool = True, trainable: bool = False) -> None:
        """
        Leaky-Integrate-and-Fire(LIF)神经元。
        一阶电位变换公式为：
        $$τ\frac{du}{dt}=-(u-u_{rest})+RI$$
        @params:
            tau_m: float 膜时间常数$τ_{m}$
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            spiking_function: Module 计算脉冲时所使用的阶跃函数
            hard_reset: bool 是否为硬重置
            trainable: bool 参数是否可以训练
        """
        super().__init__(
            tau_m = tau_m,
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            trainable = trainable
        )
        self.multi_time_step_function = multi_time_step_lif()


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return super().extra_repr() + "tau_m=%g, u_threshold=%g, u_rest=%g" % (self.tau_m, self.u_threshold, self.u_rest)


    def forward_multi_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        @params:
            x: torch.Tensor 来自突触的输入电位$X_{i}^{l}(t)$
        @return:
            o: torch.Tensor 胞体当前的输出脉冲$O_{i}^{l}(t)$
        """
        self.u = self.init_tensor(self.u, x[0])
        o, self.u = self.multi_time_step_function.apply(x, self.u, self.tau_m, self.u_threshold, self.u_rest, self.spiking_function_prototype, self.spiking_function.a, self.reset_prototype)
        return o