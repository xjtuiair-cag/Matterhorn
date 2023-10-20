# -*- coding: UTF-8 -*-
"""
脉冲神经网络神经元的胞体，一层的后半段。输入为模拟电位值，输出为脉冲。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.snn.skeleton import Module
from matterhorn_pytorch.snn import surrogate
from typing import Callable, Iterable
try:
    from rich import print
except:
    pass


class Soma(Module):
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
        super().__init__()
        self.tau_m = nn.Parameter(torch.tensor(tau_m), requires_grad = trainable)
        self.u = 0.0
        self.u_threshold = u_threshold
        self.u_rest = u_rest
        self.spiking_function = spiking_function
        self.hard_reset = hard_reset
        self.trainable = trainable
        self.reset()


    def supports_multi_time_step(self) -> bool:
        """
        是否支持多个时间步。
        @return:
            if_support: bool 是否支持多个时间步
        """
        return False


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return "multi_time_step=%s, trainable=%s, reset=%s, " % ("True" if self.multi_time_step else "False", "True" if self.trainable else "False", "\"hard\"" if self.hard_reset else "\"soft\"")


    def init_tensor(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        校正整个电位形状
        @params:
            u: torch.Tensor 待校正的电位，可能是张量或浮点值
            x: torch.Tensor 带有正确数据类型、所在设备和形状的张量
        @return:
            u: torch.Tensor 经过校正的电位张量
        """
        if isinstance(u, float):
            u = torch.full_like(x, u)
            u = u.detach().requires_grad_(True)
        return u


    def reset(self) -> None:
        """
        重置整个神经元
        """
        self.detach()
        self.u = self.u_rest

    
    def detach(self) -> None:
        """
        将历史电位从计算图中分离，以停止在时间上进行反向传播。
        """
        if isinstance(self.u, torch.Tensor):
            self.u = self.u.detach().requires_grad_(True)


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        @params:
            h: torch.Tensor 上一时刻的电位$U_{i}^{l}(t-1)$
            x: torch.Tensor 输入电位$X_{i}^{l}(t)$
        @return:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        """
        du = (1.0 / self.tau_m) * (-(h - self.u_rest) + x)
        u = h + du
        return u


    def f_firing(self, u: torch.Tensor) -> torch.Tensor:
        """
        通过当前电位$U_{i}^{l}(t)$计算当前脉冲$O_{i}^{l}(t)$。
        @params:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        @return:
            o: torch.Tensor 当前脉冲$O_{i}^{l}(t)$
        """
        return self.spiking_function(u - self.u_threshold)


    def f_reset(self, u: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """
        通过当前脉冲$O_{i}^{l}(t)$重置当前电位$U_{i}^{l}(t)$。
        @params:
            u: torch.Tensor 当前电位$U_{i}^{l}(t-1)$
            o: torch.Tensor 当前脉冲$O_{i}^{l}(t-1)$
        @return:
            h: torch.Tensor 经过重置之后的当前电位$U_{i}^{l}(t-1)$
        """
        if self.hard_reset:
            h = u * (1.0 - o) + self.u_rest * o
        else:
            h = u - (self.u_threshold - self.u_rest) * o
        return h


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
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
    

class IF(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: Module = surrogate.Rectangular(), hard_reset: bool = True) -> None:
        """
        Integrate-and-Fire(IF)神经元。
        无泄漏过程，一阶电位变换公式为：
        $$\frac{du}{dt}=RI$$
        @params:
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            spiking_function: Module 计算脉冲时所使用的阶跃函数
            hard_reset: bool 是否为硬重置
        """
        super().__init__(
            tau_m = 1.0,
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset
        )
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return super().extra_repr() + "u_th=%.3f, u_rest=%.3f" % (self.u_threshold, self.u_rest)


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        @params:
            h: torch.Tensor 上一时刻的电位$U_{i}^{l}(t-1)$
            x: torch.Tensor 输入电位$X_{i}^{l}(t)$
        @return:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        """
        u = h + x
        return u


class LIF(Soma):
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


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return super().extra_repr() + "tau_m=%.3f, u_th=%.3f, u_rest=%.3f" % (self.tau_m, self.u_threshold, self.u_rest)


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        @params:
            h: torch.Tensor 上一时刻的电位$U_{i}^{l}(t-1)$
            x: torch.Tensor 输入电位$X_{i}^{l}(t)$
        @return:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        """
        du = (1.0 / self.tau_m) * (-(h - self.u_rest) + x)
        u = h + du
        return u


class QIF(Soma):
    def __init__(self, tau_m: float = 2.0, u_threshold: float = 1.0, u_rest: float = 0.0, u_c: float = 0.8, a_0: float = 1.0, spiking_function: Module = surrogate.Rectangular(), hard_reset: bool = True, trainable: bool = False) -> None:
        """
        Quadratic Integrate-and-Fire(QIF)神经元。
        一阶电位变换公式为：
        $$τ\frac{du}{dt}=a_{0}(u-u_{rest})(u-u_{c})+RI$$
        @params:
            tau_m: float 膜时间常数$τ_{m}$
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            u_c: float 参数$u_{c}$
            a_0: float 参数$a_{0}$
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
        self.a_0 = nn.Parameter(torch.tensor(a_0), requires_grad = trainable)
        self.u_c = nn.Parameter(torch.tensor(u_c), requires_grad = trainable)
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return super().extra_repr() + "tau_m=%.3f, u_th=%.3f, u_rest=%.3f, a_0=%.3f, u_C=%.3f" % (self.tau_m, self.u_threshold, self.u_rest, self.a_0, self.u_c)


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        @params:
            h: torch.Tensor 上一时刻的电位$U_{i}^{l}(t-1)$
            x: torch.Tensor 输入电位$X_{i}^{l}(t)$
        @return:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        """
        du = (1.0 / self.tau_m) * (-self.a_0 * (h - self.u_rest) * (h - self.u_c) + x)
        u = h + du
        return u


class EIF(Soma):
    def __init__(self, tau_m: float = 2.0, u_threshold: float = 1.0, u_rest: float = 0.0, u_t: float = 8.0, delta_t: float = 1.0, spiking_function: Module = surrogate.Rectangular(), hard_reset: bool = True, trainable: bool = False) -> None:
        """
        Exponential Integrate-and-Fire(EIF)神经元。
        一阶电位变换公式为：
        $$τ\frac{du}{dt}=-(u-u_{rest})+Δ_{T}e^{\frac{u-u_{T}}{Δ_{T}}}+RI$$
        @params:
            tau_m: float 膜时间常数$τ_{m}$
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            u_t: float 参数$u_{T}$
            delta_t: float 参数$Δ_{T}$
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
        self.delta_t = nn.Parameter(torch.tensor(delta_t), requires_grad = trainable)
        self.u_t = nn.Parameter(torch.tensor(u_t), requires_grad = trainable)
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return super().extra_repr() + "tau_m=%.3f, u_th=%.3f, u_rest=%.3f, u_T=%.3f, delta_T=%.3f" % (self.tau_m, self.u_threshold, self.u_rest, self.u_t, self.delta_t)


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        @params:
            h: torch.Tensor 上一时刻的电位$U_{i}^{l}(t-1)$
            x: torch.Tensor 输入电位$X_{i}^{l}(t)$
        @return:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        """
        du = (1.0 / self.tau_m) * (-(h - self.u_rest) + self.delta_t * torch.exp((h - self.u_t) / self.delta_t) + x)
        u = h + du
        return u


class Izhikevich(Soma):
    def __init__(self, a: float = 1.0, b: float = 1.0, u_threshold: float = 1.0, spiking_function: Module = surrogate.Rectangular(), hard_reset: bool = True, trainable = False) -> None:
        """
        Izhikevich神经元。
        一阶电位变换公式为：
        $$\frac{du}{dt}=0.04u^{2}+5u+140-w+I$$
        $$\frac{dw}{dt}=a(bu-w)$$
        @params:
            tau_m: float 膜时间常数$τ_{m}$
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            u_t: float 参数$u_{T}$
            delta_t: float 参数$Δ_{T}$
            spiking_function: Module 计算脉冲时所使用的阶跃函数
            hard_reset: bool 是否为硬重置
            trainable: bool 参数是否可以训练
        """
        super().__init__(
            tau_m = 1.0,
            u_threshold = u_threshold,
            u_rest = 0.0,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            trainable = trainable
        )
        self.w = 0.0
        self.a = nn.Parameter(torch.tensor(a), requires_grad = trainable)
        self.b = nn.Parameter(torch.tensor(b), requires_grad = trainable)
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return super().extra_repr() + "a=%.3f, b=%.3f, u_th=%.3f" % (self.a, self.b, self.u_threshold)


    def reset(self) -> None:
        """
        重置整个神经元
        """
        self.detach()
        self.u = 0.0
        self.w = 0.0

    
    def detach(self) -> None:
        """
        将历史电位与权重从计算图中分离，以停止在时间上进行反向传播。
        """
        if isinstance(self.u, torch.Tensor):
            self.u = self.u.detach().requires_grad_(True)
        if isinstance(self.w, torch.Tensor):
            self.w = self.w.detach().requires_grad_(True)


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        @params:
            h: torch.Tensor 上一时刻的电位$U_{i}^{l}(t-1)$
            x: torch.Tensor 输入电位$X_{i}^{l}(t)$
        @return:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        """
        self.w = self.init_tensor(self.w, h)
        dw = self.a * (self.b * h - self.w)
        self.w = self.w + dw
        du = 0.04 * h * h + 5.0 * h + 140.0 - self.w + x
        u = h + du
        return u


class Response(Soma):
    def __init__(self, response_function: Callable, param_list: Iterable = [], u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: Module = surrogate.Rectangular(), hard_reset: bool = True, trainable: bool = False) -> None:
        """
        可以自定义反应函数的胞体。
        @params:
            response_function: Callable 自定义的反应函数，接受3个参数：历史电位$H^{l}(t)$、输入电位$X^{l}(t)$和模型的可训练参数列表
            param_list: Iterable 可训练参数列表的初始化值，类型都是Number。固定参数请固定地写在反应函数中
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            spiking_function: Module 计算脉冲时所使用的阶跃函数
            hard_reset: bool 是否为硬重置
            trainable: bool 参数是否可以训练
        """
        super().__init__(
            tau_m = 1.0,
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            trainable = trainable
        )
        self.response_function = response_function
        self.param_list = [nn.Parameter(torch.tensor(x), requires_grad = trainable) for x in param_list]


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        @params:
            h: torch.Tensor 上一时刻的电位$U_{i}^{l}(t-1)$
            x: torch.Tensor 输入电位$X_{i}^{l}(t)$
        @return:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        """
        return self.response_function(h, x, self.param_list)


class AnalogSoma(Soma):
    def __init__(self, tau_m: float = 1.0, u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: Module = surrogate.Rectangular(), activation_function: nn.Module = nn.ReLU(), hard_reset: bool = True, trainable: bool = False) -> None:
        """
        带有模拟输出的Response-Firing-Reset三段式神经元胞体骨架，分别为：
        （1）通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$；
        （2）通过当前电位$U_{i}^{l}(t)$计算当前脉冲$O_{i}^{l}(t)$；
        （3）通过当前电位$U_{i}^{l}(t)$计算当前模拟输出$Y_{i}^{l}(t)$；
        （4）通过当前脉冲$O_{i}^{l}(t)$重置当前电位$U_{i}^{l}(t)$。
        @params:
            tau_m: float 膜时间常数$τ_{m}$
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            spiking_function: Module 计算脉冲时所使用的阶跃函数
            activation_function: nn.Module 激活函数
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
        self.activation_function = activation_function
    

    def f_activation(self, u: torch.Tensor) -> torch.Tensor:
        """
        通过当前电位$U_{i}^{l}(t)$计算当前模拟输出$Y_{i}^{l}(t)$。
        @params:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        @return:
            y: torch.Tensor 当前模拟输出$Y_{i}^{l}(t)$
        """
        return self.activation_function(u - self.u_rest)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            x: torch.Tensor 来自突触的输入电位$X_{i}^{l}(t)$
        @return:
            o: torch.Tensor 胞体当前的输出脉冲$O_{i}^{l}(t)$
        """
        self.u = self.init_tensor(self.u, x)
        self.u = self.f_response(self.u, x)
        o = self.f_firing(self.u)
        y = self.f_activation(self.u)
        self.u = self.f_reset(self.u, o)
        return y


class KLIF(AnalogSoma):
    def __init__(self, tau_m: float = 1, u_threshold: float = 1, u_rest: float = 0, k: float = 0.2, hard_reset: bool = True, trainable: bool = False) -> None:
        """
        KLIF神经元
        @params:
            tau_m: float 膜时间常数$τ_{m}$
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            k: float 参数k
            hard_reset: bool 是否为硬重置
            trainable: bool 参数是否可以训练
        """
        super().__init__(
            tau_m = tau_m,
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = self.f_kspiking,
            activation_function = self.f_kspiking,
            hard_reset = hard_reset,
            trainable = trainable
        )
        self.k = nn.Parameter(torch.tensor(k), requires_grad = trainable)
        

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return super().extra_repr() + "tau_m=%.3f, u_th=%.3f, u_rest=%.3f" % (self.tau_m, self.u_threshold, self.u_rest)


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        @params:
            h: torch.Tensor 上一时刻的电位$U_{i}^{l}(t-1)$
            x: torch.Tensor 输入电位$X_{i}^{l}(t)$
        @return:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        """
        du = (1.0 / self.tau_m) * (-(h - self.u_rest) + x)
        u = h + du
        return u


    def f_kspiking(self, u: torch.Tensor) -> torch.Tensor:
        """
        通过当前电位$U_{i}^{l}(t)$计算当前脉冲$O_{i}^{l}(t)$。
        @params:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        @return:
            o: torch.Tensor 当前脉冲$O_{i}^{l}(t)$
        """
        return nn.functional.relu(self.k * u)


class LIAF(AnalogSoma):
    def __init__(self, tau_m: float = 1.0, u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: Module = surrogate.Rectangular(), activation_function: nn.Module = nn.ReLU(), hard_reset: bool = True, trainable: bool = False) -> None:
        """
        Leaky Integrate-and-Analog-Fire(LIAF)神经元
        @params:
            tau_m: float 膜时间常数$τ_{m}$
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            spiking_function: Module 计算脉冲时所使用的阶跃函数
            activation_function: nn.Module 激活函数
            hard_reset: bool 是否为硬重置
            trainable: bool 参数是否可以训练
        """
        super().__init__(
            tau_m = tau_m,
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            activation_function = activation_function,
            hard_reset = hard_reset,
            trainable = trainable
        )
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return super().extra_repr() + "tau_m=%.3f, u_th=%.3f, u_rest=%.3f" % (self.tau_m, self.u_threshold, self.u_rest)


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        @params:
            h: torch.Tensor 上一时刻的电位$U_{i}^{l}(t-1)$
            x: torch.Tensor 输入电位$X_{i}^{l}(t)$
        @return:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        """
        du = (1.0 / self.tau_m) * (-(h - self.u_rest) + x)
        u = h + du
        return u