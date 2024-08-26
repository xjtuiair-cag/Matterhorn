# -*- coding: UTF-8 -*-
"""
脉冲神经网络神经元的胞体，一层的后半段。输入为模拟电位值，输出为脉冲。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
"""


import torch
import torch.nn as nn
import torch.nn.functional as _F
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME
import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.skeleton import Module as _Module
from matterhorn_pytorch.snn.firing import Firing as _Firing, Gaussian as _Gaussian
from matterhorn_pytorch.snn.soma_functional import *
from torch.utils.cpp_extension import load_inline
import matterhorn_pytorch._ext.cpp as _ext_cpp
import matterhorn_pytorch._ext.cuda as _ext_cu
from typing import Any as _Any, Mapping as _Mapping, Callable as _Callable
import warnings
from subprocess import SubprocessError


_EXT_DEBUG_MODE = False
warnings.simplefilter('once', UserWarning)


class Soma(_Module):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, enable_exts: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        Response-Firing-Reset三段式神经元胞体骨架，分别为：
        （1）通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$；
        （2）通过当前电位$U_{i}^{l}(t)$计算当前脉冲$O_{i}^{l}(t)$；
        （3）通过当前脉冲$O_{i}^{l}(t)$重置当前电位$U_{i}^{l}(t)$。
        Args:
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            spiking_function (Firing): 计算脉冲时所使用的阶跃函数
            hard_reset (bool): 是否为硬重置
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__()
        self.register_buffer("u", None)
        self.u_threshold = nn.Parameter(torch.tensor(u_threshold, device = device, dtype = dtype), requires_grad = False)
        self.u_rest = nn.Parameter(torch.tensor(u_rest, device = device, dtype = dtype), requires_grad = False)
        self.spiking_function: _Firing = spiking_function
        self.hard_reset = hard_reset
        self.enable_exts = enable_exts


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        exts = self.exts if self.enable_exts else dict()
        return ", ".join(["u_threshold=%g" % self.u_threshold, "u_rest=%g" % self.u_rest, "reset=%s" % ('"zero"' if self.hard_reset else '"sub"',)]) + ((", exts=" + repr(list(exts.keys()))) if len(exts.keys()) else "") + ((", " + super().extra_repr()) if len(super().extra_repr()) else "")


    def multi_step_mode_(self, if_on: bool = True, recursive: bool = True) -> nn.Module:
        """
        调整模型至多时间步模式。
        Args:
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
            recursive (bool): 是否递归调整子模块的时间步模式
        """
        return super().multi_step_mode_(if_on, recursive = False)


    def _check_buffer(self, x: torch.Tensor) -> _Module:
        """
        检查临时变量。
        Args:
            x (torch.Tensor): 关键张量
        """
        self.detach()
        if self.u is None or (isinstance(self.u, torch.Tensor) and self.u.shape != x.shape):
            self.u = torch.full_like(x, self.u_rest)
        return self


    def reset(self) -> _Module:
        """
        重置整个神经元。
        """
        self.detach()
        super().reset()
        if self.u is not None:
            self.u = torch.full_like(self.u, self.u_rest)
        return self

    
    def detach(self) -> _Module:
        """
        将历史电位从计算图中分离，以停止在时间上进行反向传播。
        """
        super().detach()
        if isinstance(self.u, torch.Tensor):
            self.u = self.u.detach()
        return self


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        Args:
            h (torch.Tensor): 上一时刻的电位$U_{i}^{l}(t-1)$
            x (torch.Tensor): 输入电位$X_{i}^{l}(t)$
        Returns:
            u (torch.Tensor): 当前电位$U_{i}^{l}(t)$
        """
        pass


    def f_firing(self, u: torch.Tensor) -> torch.Tensor:
        """
        通过当前电位$U_{i}^{l}(t)$计算当前脉冲$O_{i}^{l}(t)$。
        Args:
            u (torch.Tensor): 当前电位$U_{i}^{l}(t)$
        Returns:
            o (torch.Tensor): 当前脉冲$O_{i}^{l}(t)$
        """
        return self.spiking_function(u, self.u_threshold, self.u_rest)


    def f_reset(self, u: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """
        通过当前脉冲$O_{i}^{l}(t)$重置当前电位$U_{i}^{l}(t)$。
        Args:
            u (torch.Tensor): 当前电位$U_{i}^{l}(t-1)$
            o (torch.Tensor): 当前脉冲$O_{i}^{l}(t-1)$
        Returns:
            h (torch.Tensor): 经过重置之后的当前电位$U_{i}^{l}(t-1)$
        """
        if self.hard_reset:
            o = oo.apply(o)
            h = u * (1.0 - o) + self.u_rest * o
        else:
            h = u - (self.u_threshold - self.u_rest) * o
        return h


    def build_ext(self, ext_name: str, **kwargs) -> _Any:
        """
        构建单个扩展。
        Args:
            ext_name (str): 扩展名
            **kwargs (str: Any): 构建参数
        """
        res = None
        try:
            kwargs["verbose"] = _EXT_DEBUG_MODE
            res = load_inline(**kwargs)
        except Exception as e:
            if _EXT_DEBUG_MODE:
                warnings.warn("Failed to compile %s extensions. %s" % (ext_name, str(e).split("\n")[0]))
        return res


    @property
    def exts(self) -> _Mapping[str, object]:
        """
        构建扩展。
        """
        return dict()


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 来自突触的输入电位$X_{i}^{l}(t)$
        Returns:
            o (torch.Tensor): 胞体当前的输出脉冲$O_{i}^{l}(t)$
        """
        self._check_buffer(x)
        self.u = self.f_response(self.u, x)
        o = self.f_firing(self.u)
        self.u = self.f_reset(self.u, o)
        return o


    def forward_steps_on_ext(self, x: torch.Tensor, exts: _Mapping[str, object], ext_name: str) -> torch.Tensor:
        """
        多个时间步的前向传播函数（基于扩展）。
        Args:
            x (torch.Tensor): 来自突触的输入电位$X_{i}^{l}(t)$
        Returns:
            o (torch.Tensor): 胞体当前的输出脉冲$O_{i}^{l}(t)$
        """
        return super().forward_steps(x)


    def forward_steps(self, x: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 来自突触的输入电位$X_{i}^{l}(t)$
        Returns:
            o (torch.Tensor): 胞体当前的输出脉冲$O_{i}^{l}(t)$
        """
        device: torch.device = x.device
        if self.enable_exts:
            exts = self.exts
            if device.type == "cuda" and "cuda" in exts:
                return self.forward_steps_on_ext(x, exts, "cuda")
            if device.type == "cpu" and "cpp" in exts:
                return self.forward_steps_on_ext(x, exts, "cpp")
        return super().forward_steps(x)


class IF(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, enable_exts: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        Integrate-and-Fire(IF)神经元。
        无泄漏过程，一阶电位变换公式为：
        $$\frac{du}{dt}=RI$$
        Args:
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            spiking_function (Firing): 计算脉冲时所使用的阶跃函数
            hard_reset (bool): 是否为硬重置
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            enable_exts = enable_exts,
            device = device,
            dtype = dtype
        )


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        Args:
            h (torch.Tensor): 上一时刻的电位$U_{i}^{l}(t-1)$
            x (torch.Tensor): 输入电位$X_{i}^{l}(t)$
        Returns:
            u (torch.Tensor): 当前电位$U_{i}^{l}(t)$
        """
        u = h + x
        return u


class LIF(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, tau_m: float = 2.0, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, trainable: bool = False, enable_exts: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
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
            trainable (bool): 参数是否可以训练
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            enable_exts = enable_exts,
            device = device,
            dtype = dtype
        )
        self.tau_m = nn.Parameter(torch.tensor(tau_m, device = device, dtype = dtype), requires_grad = trainable)


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["tau_m=%g" % self.tau_m]) + ((", " + super().extra_repr()) if len(super().extra_repr()) else "")


    @property
    def exts(self) -> None:
        """
        构建扩展。
        """
        res = dict()
        fp_name, fp_source = _ext_cpp.fp_lif_source(self.spiking_function, self.hard_reset)
        bp_name, bp_source = _ext_cpp.bp_lif_source(self.spiking_function, self.hard_reset)
        cpp_ext = self.build_ext(
            "cpp",
            name = _ext_cpp.purify_name("lif_%s_%s_%s" % (self.spiking_function.__class__.__name__, self.spiking_function.extra_repr(), "zero" if self.hard_reset else "sub")),
            cpp_sources = [fp_source, bp_source],
            functions = [fp_name, bp_name],
            extra_cflags = ["-g", "-w"]
        )
        if cpp_ext is not None:
            res["cpp"] = cpp_ext
        if torch.cuda.is_available():
            fp_name, fp_dec, fp_source = _ext_cu.fp_lif_source(self.spiking_function, self.hard_reset)
            bp_name, bp_dec, bp_source = _ext_cu.bp_lif_source(self.spiking_function, self.hard_reset)
            cuda_ext = self.build_ext(
                "cuda",
                name = _ext_cu.purify_name("lif_cu_%s_%s_%s" % (self.spiking_function.__class__.__name__, self.spiking_function.extra_repr(), "zero" if self.hard_reset else "sub")),
                cpp_sources = [fp_dec, bp_dec],
                cuda_sources = [fp_source, bp_source],
                functions = [fp_name, bp_name],
                extra_cflags = ["-g", "-w"]
            )
            if cuda_ext is not None:
                res["cuda"] = cuda_ext
        return res


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        Args:
            h (torch.Tensor): 上一时刻的电位$U_{i}^{l}(t-1)$
            x (torch.Tensor): 输入电位$X_{i}^{l}(t)$
        Returns:
            u (torch.Tensor): 当前电位$U_{i}^{l}(t)$
        """
        du = (1.0 / self.tau_m) * (-(h - self.u_rest) + x)
        u = h + du
        return u


    def forward_steps_on_ext(self, x: torch.Tensor, exts: _Mapping[str, object], ext_name: str) -> torch.Tensor:
        """
        多个时间步的前向传播函数（基于扩展）。
        Args:
            x (torch.Tensor): 来自突触的输入电位$X_{i}^{l}(t)$
        Returns:
            o (torch.Tensor): 胞体当前的输出脉冲$O_{i}^{l}(t)$
        """
        self._check_buffer(x)
        if ext_name == "cpp":
            fp = exts["cpp"].fp_lif
            bp = exts["cpp"].bp_lif
            o, self.u = multi_step_mode_lif.apply(x, self.u, self.u_threshold, self.u_rest, self.tau_m, fp, bp)
        elif ext_name == "cuda":
            fp = exts["cuda"].fp_lif_cuda
            bp = exts["cuda"].bp_lif_cuda
            o, self.u = multi_step_mode_lif.apply(x, self.u, self.u_threshold, self.u_rest, self.tau_m, fp, bp)
        else:
            o = super().forward_steps(x)
        return o


class QIF(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, tau_m: float = 2.0, u_c: float = 1.0, a_0: float = 1.0, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, trainable: bool = False, enable_exts: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
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
            trainable (bool): 参数是否可以训练
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            enable_exts = enable_exts,
            device = device,
            dtype = dtype
        )
        self.tau_m = nn.Parameter(torch.tensor(tau_m, device = device, dtype = dtype), requires_grad = trainable)
        self.a_0 = nn.Parameter(torch.tensor(a_0, device = device, dtype = dtype), requires_grad = trainable)
        self.u_c = nn.Parameter(torch.tensor(u_c, device = device, dtype = dtype), requires_grad = trainable)
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["tau_m=%g" % self.tau_m, "a_0=%g" % self.a_0, "u_C=%g" % self.u_c]) + ((", " + super().extra_repr()) if len(super().extra_repr()) else "")


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        Args:
            h (torch.Tensor): 上一时刻的电位$U_{i}^{l}(t-1)$
            x (torch.Tensor): 输入电位$X_{i}^{l}(t)$
        Returns:
            u (torch.Tensor): 当前电位$U_{i}^{l}(t)$
        """
        du = (1.0 / self.tau_m) * (self.a_0 * (h - self.u_rest) * (h - self.u_c) + x)
        u = h + du
        return u


class ExpIF(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, tau_m: float = 2.0, u_t: float = 0.0, delta_t: float = 0.001, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, trainable: bool = False, enable_exts: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
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
            trainable (bool): 参数是否可以训练
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            enable_exts = enable_exts,
            device = device,
            dtype = dtype
        )
        self.tau_m = nn.Parameter(torch.tensor(tau_m, device = device, dtype = dtype), requires_grad = trainable)
        self.delta_t = nn.Parameter(torch.tensor(delta_t, device = device, dtype = dtype), requires_grad = trainable)
        self.u_t = nn.Parameter(torch.tensor(u_t, device = device, dtype = dtype), requires_grad = trainable)
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["tau_m=%g" % self.tau_m, "u_t=%g" % self.u_t, "delta_t=%g" % self.delta_t]) + ((", " + super().extra_repr()) if len(super().extra_repr()) else "")


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        Args:
            h (torch.Tensor): 上一时刻的电位$U_{i}^{l}(t-1)$
            x (torch.Tensor): 输入电位$X_{i}^{l}(t)$
        Returns:
            u (torch.Tensor): 当前电位$U_{i}^{l}(t)$
        """
        du = (1.0 / self.tau_m) * (-(h - self.u_rest) + self.delta_t * torch.exp((h - self.u_t) / self.delta_t) + x)
        u = h + du
        return u


class Izhikevich(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, a: float = 1.0, b: float = 1.0, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, trainable: bool = False, enable_exts: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
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
            trainable (bool): 参数是否可以训练
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            enable_exts = enable_exts,
            device = device,
            dtype = dtype
        )
        self.register_buffer("w", None)
        self.a = nn.Parameter(torch.tensor(a, device = device, dtype = dtype), requires_grad = trainable)
        self.b = nn.Parameter(torch.tensor(b, device = device, dtype = dtype), requires_grad = trainable)
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["a=%g" % self.a, "b=%g" % self.b]) + ((", " + super().extra_repr()) if len(super().extra_repr()) else "")


    def _check_buffer(self, x: torch.Tensor) -> _Module:
        """
        检查临时变量。
        Args:
            x (torch.Tensor): 关键张量
        """
        self.detach()
        super()._check_buffer(x)
        if self.w is None or (isinstance(self.u, torch.Tensor) and self.u.shape != x.shape):
            self.w = torch.full_like(x, 0.0)
        return self


    def reset(self) -> _Module:
        """
        重置整个神经元
        """
        self.detach()
        super().reset()
        if self.w is not None:
            self.w = torch.full_like(self.w, 0.0)
        return self


    def detach(self) -> _Module:
        """
        将历史电位与权重从计算图中分离，以停止在时间上进行反向传播。
        """
        super().detach()
        if isinstance(self.w, torch.Tensor):
            self.w = self.w.detach()
        return self


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        Args:
            h (torch.Tensor): 上一时刻的电位$U_{i}^{l}(t-1)$
            x (torch.Tensor): 输入电位$X_{i}^{l}(t)$
        Returns:
            u (torch.Tensor): 当前电位$U_{i}^{l}(t)$
        """
        dw = self.a * (self.b * h - self.w)
        self.w = self.w + dw
        du = 0.00004 * h * h + 0.005 * h + 0.14 + self.u_rest - self.w + x
        u = h + du
        return u


class KLIF(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, tau_m: float = 2.0, k: float = 0.2, spiking_function: _Firing = _Gaussian(), hard_reset: bool = True, trainable: bool = False, enable_exts: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        KLIF神经元
        Args:
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            tau_m (float): 膜时间常数$τ_{m}$
            k (float): 参数k
            spiking_function (Firing): 计算脉冲时所使用的阶跃函数
            hard_reset (bool): 是否为硬重置
            trainable (bool): 参数是否可以训练
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            enable_exts = enable_exts,
            device = device,
            dtype = dtype
        )
        self.tau_m = nn.Parameter(torch.tensor(tau_m, device = device, dtype = dtype), requires_grad = trainable)
        self.k = nn.Parameter(torch.tensor(k, device = device, dtype = dtype), requires_grad = trainable)


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["tau_m=%g" % self.tau_m, "k=%g" % self.k]) + ((", " + super().extra_repr()) if len(super().extra_repr()) else "")


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        Args:
            h (torch.Tensor): 上一时刻的电位$U_{i}^{l}(t-1)$
            x (torch.Tensor): 输入电位$X_{i}^{l}(t)$
        Returns:
            u (torch.Tensor): 当前电位$U_{i}^{l}(t)$
        """
        du = (1.0 / self.tau_m) * (-(h - self.u_rest) + x)
        u = h + du
        f = _F.relu(self.k * (u - self.u_rest)) + self.u_rest
        return f


class AnalogSoma(Soma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: _Firing = _Gaussian(), activation_function: nn.Module = nn.ReLU(), hard_reset: bool = True, enable_exts: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        带有模拟输出的Response-Firing-Reset三段式神经元胞体骨架，分别为：
        （1）通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$；
        （2）通过当前电位$U_{i}^{l}(t)$计算当前脉冲$O_{i}^{l}(t)$；
        （3）通过当前电位$U_{i}^{l}(t)$计算当前模拟输出$Y_{i}^{l}(t)$；
        （4）通过当前脉冲$O_{i}^{l}(t)$重置当前电位$U_{i}^{l}(t)$。
        Args:
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            spiking_function (Firing): 计算脉冲时所使用的阶跃函数
            activation_function (nn.Module): 激活函数
            hard_reset (bool): 是否为硬重置
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            hard_reset = hard_reset,
            enable_exts = enable_exts,
            device = device,
            dtype = dtype
        )
        self.activation_function = activation_function
    

    def f_activation(self, u: torch.Tensor) -> torch.Tensor:
        """
        通过当前电位$U_{i}^{l}(t)$计算当前模拟输出$Y_{i}^{l}(t)$。
        Args:
            u (torch.Tensor): 当前电位$U_{i}^{l}(t)$
        Returns:
            y (torch.Tensor): 当前模拟输出$Y_{i}^{l}(t)$
        """
        return self.activation_function(u - self.u_rest)
    

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 来自突触的输入电位$X_{i}^{l}(t)$
        Returns:
            y (torch.Tensor): 当前模拟输出$Y_{i}^{l}(t)$
        """
        self._check_buffer(x)
        self.u = self.f_response(self.u, x)
        o = self.f_firing(self.u)
        y = self.f_activation(self.u)
        self.u = self.f_reset(self.u, o)
        return y


class LIAF(AnalogSoma):
    def __init__(self, u_threshold: float = 1.0, u_rest: float = 0.0, tau_m: float = 2.0, spiking_function: _Firing = _Gaussian(), activation_function: nn.Module = nn.ReLU(), hard_reset: bool = True, trainable: bool = False, enable_exts: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        Leaky Integrate-and-Analog-Fire(LIAF)神经元
        Args:
            u_threshold (float): 阈电位$u_{th}$
            u_rest (float): 静息电位$u_{rest}$
            tau_m (float): 膜时间常数$τ_{m}$
            spiking_function (Firing): 计算脉冲时所使用的阶跃函数
            activation_function (nn.Module): 激活函数
            hard_reset (bool): 是否为硬重置
            trainable (bool): 参数是否可以训练
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            u_threshold = u_threshold,
            u_rest = u_rest,
            spiking_function = spiking_function,
            activation_function = activation_function,
            hard_reset = hard_reset,
            enable_exts = enable_exts,
            device = device,
            dtype = dtype
        )
        self.tau_m = nn.Parameter(torch.tensor(tau_m, device = device, dtype = dtype), requires_grad = trainable)


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["tau_m=%g" % self.tau_m]) + ((", " + super().extra_repr()) if len(super().extra_repr()) else "")


    @property
    def exts(self) -> None:
        """
        构建扩展。
        """
        res = dict()
        fp_name, fp_source = _ext_cpp.fp_lif_source(self.spiking_function, self.hard_reset)
        bp_name, bp_source = _ext_cpp.bp_lif_source(self.spiking_function, self.hard_reset)
        cpp_ext = self.build_ext(
            "cpp",
            name = _ext_cpp.purify_name("lif_%s_%s_%s" % (self.spiking_function.__class__.__name__, self.spiking_function.extra_repr(), "zero" if self.hard_reset else "sub")),
            cpp_sources = [fp_source, bp_source],
            functions = [fp_name, bp_name],
            extra_cflags = ["-g", "-w"]
        )
        if cpp_ext is not None:
            res["cpp"] = cpp_ext
        if torch.cuda.is_available():
            fp_name, fp_dec, fp_source = _ext_cu.fp_lif_source(self.spiking_function, self.hard_reset)
            bp_name, bp_dec, bp_source = _ext_cu.bp_lif_source(self.spiking_function, self.hard_reset)
            cuda_ext = self.build_ext(
                "cuda",
                name = _ext_cu.purify_name("lif_cu_%s_%s_%s" % (self.spiking_function.__class__.__name__, self.spiking_function.extra_repr(), "zero" if self.hard_reset else "sub")),
                cpp_sources = [fp_dec, bp_dec],
                cuda_sources = [fp_source, bp_source],
                functions = [fp_name, bp_name],
                extra_cflags = ["-g", "-w"]
            )
            if cuda_ext is not None:
                res["cuda"] = cuda_ext
        return res


    def f_response(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的电位$U_{i}^{l}(t-1)$和当前时刻的输入电位$X_{i}^{l}(t)$计算电位导数$dU/dt=U_{i}^{l}(t)-U_{i}^{l}(t-1)$，进而获得当前电位$U_{i}^{l}(t)$。
        Args:
            h (torch.Tensor): 上一时刻的电位$U_{i}^{l}(t-1)$
            x (torch.Tensor): 输入电位$X_{i}^{l}(t)$
        Returns:
            u (torch.Tensor): 当前电位$U_{i}^{l}(t)$
        """
        du = (1.0 / self.tau_m) * (-(h - self.u_rest) + x)
        u = h + du
        return u


    def forward_steps_on_ext(self, x: torch.Tensor, exts: _Mapping[str, object], ext_name: str) -> torch.Tensor:
        """
        多个时间步的前向传播函数（基于扩展）。
        Args:
            x (torch.Tensor): 来自突触的输入电位$X_{i}^{l}(t)$
        Returns:
            y (torch.Tensor): 当前模拟输出$Y_{i}^{l}(t)$
        """
        self._check_buffer(x)
        if ext_name == "cpp":
            fp = exts["cpp"].fp_lif
            bp = exts["cpp"].bp_lif
            u, self.u = multi_step_mode_liaf.apply(x, self.u, self.u_threshold, self.u_rest, self.tau_m, fp, bp)
            y = self.activation_function(u - self.u_rest)
            print(u)
        elif ext_name == "cuda":
            fp = exts["cuda"].fp_lif_cuda
            bp = exts["cuda"].bp_lif_cuda
            u, self.u = multi_step_mode_liaf.apply(x, self.u, self.u_threshold, self.u_rest, self.tau_m, fp, bp)
            y = self.activation_function(u - self.u_rest)
        else:
            y = super().forward_steps(x)
        return y