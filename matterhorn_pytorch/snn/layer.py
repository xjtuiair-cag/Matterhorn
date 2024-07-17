# -*- coding: UTF-8 -*-
"""
脉冲神经网络的整个神经元层，输入为脉冲，输出为脉冲。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
"""


import torch
import torch.nn as nn
import torch.nn.functional as _F
import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.skeleton import Module as _Module
from matterhorn_pytorch.snn.soma import Soma as _Soma
from matterhorn_pytorch.snn.container import Temporal as _Temporal
from typing import Any as _Any, Tuple as _Tuple, Iterable as _Iterable, Callable as _Callable, Optional as _Optional, Union as _Union
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from torch.types import _size
from matterhorn_pytorch.training.functional import stdp_online as _stdp_online


class Layer(_Module):
    def __init__(self, spike_mode: str = "s") -> None:
        """
        突触函数的骨架，定义突触最基本的函数。
        Args:
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        super().__init__()
        self.spike_mode = spike_mode.lower() if isinstance(spike_mode, str) else None


    def forward_steps(self, *args, **kwargs) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            *args: 输入
            **kwargs: 输入
        Returns:
            res (torch.Tensor): 输出
        """
        args, kwargs, tb = _SF.merge_time_steps_batch_size(args, kwargs)
        res = self.forward_step(*args, **kwargs)
        res = _SF.split_time_steps_batch_size(res, tb)
        return res


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            *args: 输入
            **kwargs: 输入
        Returns:
            res (torch.Tensor): 输出
        """
        res = super().forward(*args, **kwargs)
        if self.spike_mode in ("s", "m"):
            f = _SF.floor if self.spike_mode == "m" else _SF.val_to_spike
            if isinstance(res, _Tuple):
                res = (f(y) if isinstance(y, torch.Tensor) else y for y in res)
            else:
                res = f(res)
        return res


class STDPLayer(_Module):
    def __init__(self, a_pos: float = 0.015, tau_pos: float = 2.0, a_neg: float = 0.015, tau_neg: float = 2.0) -> None:
        """
        含有STDP学习机制的层。
        Args:
            a_pos (float): STDP参数A+
            tau_pos (float): STDP参数tau+
            a_neg (float): STDP参数A-
            tau_neg (float): STDP参数tau-
        """
        super().__init__()
        self.input_trace = None
        self.output_trace = None
        self.a_pos = a_pos
        self.tau_pos = tau_pos
        self.a_neg = a_neg
        self.tau_neg = tau_neg


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["pos=%g*exp(-x/%g)" % (self.a_pos, self.tau_pos), "neg=-%g*exp(x/%g)" % (self.a_neg, self.tau_neg)]) + (", " + super().extra_repr(self) if len(super().extra_repr(self)) else "")


    def reset(self) -> nn.Module:
        """
        重置模型。
        """
        self.input_trace = None
        self.output_trace = None
        return super().reset()


    def check_if_reset(self, u: torch.Tensor, x: torch.Tensor) -> None:
        """
        检查张量是否（在形状上）一致。若不一致，重置胞体。
        Args:
            u (torch.Tensor): 待检测张量
            x (torch.Tensor): 需要保持一致的张量
        """
        if isinstance(u, torch.Tensor) and u.shape != x.shape:
            self.reset()


class f_stdp_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, input: torch.Tensor, weight: torch.Tensor, input_trace: torch.Tensor, output_trace: torch.Tensor, soma: _Callable, a_pos: float = 0.015, tau_pos: float = 2.0, a_neg: float = 0.015, tau_neg: float = 2.0, training: bool = True, multi_step_mode: bool = True) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        利用STDP进行学习的全连接层的前向传播函数。
        Args:
            ctx (Any): 上下文
            input (torch.Tensor): 输入脉冲序列
            weight (torch.Tensor): 权重矩阵
            input_trace (torch.Tensor): 输入的迹，累积的输入效应
            output_trace (torch.Tensor): 输出的迹，累积的输出效应
            soma (snn.Module): 胞体
            a_pos (float): STDP参数A+
            tau_pos (float): STDP参数tau+
            a_neg (float): STDP参数A-
            tau_neg (float): STDP参数tau-
            training (bool): 是否正在训练
            multi_step_mode (bool): 是否为多时间步模式
        Returns:
            output (torch.Tensor): 输出脉冲序列
            input_trace (torch.Tensor): 输入的迹，累积的输入效应
            output_trace (torch.Tensor): 输出的迹，累积的输出效应
        """
        if multi_step_mode:
            flattened_input, _, tb = _SF.merge_time_steps_batch_size(input)
            psp = _F.linear(flattened_input[0], weight, bias = None)
            psp = _SF.split_time_steps_batch_size(psp, tb)
        else:
            psp = _F.linear(input, weight, bias = None)
        output: torch.Tensor = soma(psp)
        if multi_step_mode:
            input_spike_train = input.clone()
            output_spike_train = output.clone()
        else:
            input_spike_train = input[None]
            output_spike_train = output[None]
        delta_weight = torch.zeros_like(weight)
        if training:
            T, B, C = input_spike_train.shape
            T, B, N = output_spike_train.shape
            N, C = weight.shape
            for t in range(T):
                delta_weight, input_trace, output_trace = _stdp_online(
                    delta_weight = delta_weight, # [O, I]
                    input_trace = input_trace, # [B, O, I]
                    output_trace = output_trace, # [B, O, I]
                    input_spike_train = input_spike_train[t], # [B, I]
                    output_spike_train = output_spike_train[t], # [B, O]
                    a_pos = a_pos,
                    tau_pos = tau_pos,
                    a_neg = a_neg,
                    tau_neg = tau_neg
                )
        ctx.save_for_backward(delta_weight, input)
        return output, input_trace, output_trace


    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor, grad_input_trace: torch.Tensor, grad_output_trace: torch.Tensor) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None, None]:
        """
        利用STDP进行学习的全连接层的反向传播函数。
        Args:
            ctx (Any): 上下文
            grad_output (torch.Tensor): 输出脉冲序列梯度
            grad_input_trace (torch.Tensor): 输入的迹梯度
            grad_output_trace (torch.Tensor): 输出的迹梯度
        Returns:
            grad_input (torch.Tensor): 输入脉冲序列梯度
            grad_weight (torch.Tensor): 权重矩阵梯度
            grad_input_trace (torch.Tensor): 输入的迹梯度
            grad_output_trace (torch.Tensor): 输出的迹梯度
            grad_soma (None): 胞体的梯度，为None
            grad_a_pos (None): STDP参数A+的梯度，为None
            grad_tau_pos (None): STDP参数tau+的梯度，为None
            grad_a_neg (None): STDP参数A-的梯度，为None
            grad_tau_neg (None): STDP参数tau-的梯度，为None
            grad_training (None): 是否正在训练的梯度，为None
            grad_multi_step_mode (None): 是否为多时间步模式的梯度，为None
        """
        delta_weight, input = ctx.saved_tensors
        delta_weight = -delta_weight
        return torch.zeros_like(input), delta_weight, torch.zeros_like(grad_input_trace), torch.zeros_like(grad_output_trace), None, None, None, None, None, None, None


class STDPLinear(STDPLayer):
    def __init__(self, soma: _Soma, in_features: int, out_features: int, a_pos: float = 0.015, tau_pos: float = 2.0, a_neg: float = 0.015, tau_neg: float = 2.0, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        使用STDP学习机制时的全连接层。
        Args:
            soma (nn.Module): 使用的脉冲神经元胞体，在matterhorn_pytorch.snn.soma中选择
            in_features (int): 输入长度，用法同nn.Linear
            out_features (int): 输出长度，用法同nn.Linear
            a_pos (float): STDP参数A+
            tau_pos (float): STDP参数tau+
            a_neg (float): STDP参数A-
            tau_neg (float): STDP参数tau-
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            a_pos = a_pos,
            tau_pos = tau_pos,
            a_neg = a_neg,
            tau_neg = tau_neg
        )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device = device, dtype = dtype))
        nn.init.kaiming_uniform_(self.weight, a = 5.0 ** 0.5)
        self.soma = soma
        self.reset()


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["in_features=%d" % self.in_features, "out_features=%d" % self.out_features]) + (", " + super().extra_repr(self) if len(super().extra_repr(self)) else "")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        if self.multi_step_mode:
            T, B, C = x.shape
        else:
            T = 1
            B, C = x.shape
        N, C = self.weight.shape
        input_trace_shape = torch.zeros(B, N, C).to(x)
        output_trace_shape = torch.zeros(B, N, C).to(x)
        self.check_if_reset(self.input_trace, input_trace_shape)
        self.check_if_reset(self.output_trace, output_trace_shape)
        self.input_trace = _SF.to(self.input_trace, input_trace_shape)
        self.output_trace = _SF.to(self.output_trace, output_trace_shape)
        soma = self.soma.forward_steps if self.multi_step_mode else self.soma.forward_step
        y, self.input_trace, self.output_trace = f_stdp_linear.apply(x, self.weight, self.input_trace, self.output_trace, soma, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg, self.training, self.multi_step_mode)
        return y


class f_stdp_conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, input: torch.Tensor, weight: torch.Tensor, input_trace: torch.Tensor, output_trace: torch.Tensor, soma: _Callable, stride: _size_any_t, padding: _size_any_t, dilation: _size_any_t, a_pos: float = 0.015, tau_pos: float = 2.0, a_neg: float = 0.015, tau_neg: float = 2.0, training: bool = True, multi_step_mode: bool = True) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        利用STDP进行学习的2维卷积层的前向传播函数。
        Args:
            ctx (Any): 上下文
            input (torch.Tensor): 输入脉冲序列
            weight (torch.Tensor): 权重矩阵
            input_trace (torch.Tensor): 输入的迹，累积的输入效应
            output_trace (torch.Tensor): 输出的迹，累积的输出效应
            soma (snn.Module): 胞体
            stride (size_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_t): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_t): 卷积的输入步长
            a_pos (float): STDP参数A+
            tau_pos (float): STDP参数tau+
            a_neg (float): STDP参数A-
            tau_neg (float): STDP参数tau-
            training (bool): 是否正在训练
            multi_step_mode (bool): 是否为多时间步模式
        Returns:
            output (torch.Tensor): 输出脉冲序列
            input_trace (torch.Tensor): 输入的迹，累积的输入效应
            output_trace (torch.Tensor): 输出的迹，累积的输出效应
        """
        if multi_step_mode:
            flattened_input, _, tb = _SF.merge_time_steps_batch_size(input)
            psp = _F.conv2d(flattened_input[0], weight, bias = None, stride = tuple(stride), padding = tuple(padding), dilation = tuple(dilation))
            psp = _SF.split_time_steps_batch_size(psp, tb)
        else:
            psp = _F.conv2d(input, weight, bias = None, stride = tuple(stride), padding = tuple(padding), dilation = tuple(dilation))
        output: torch.Tensor = soma(psp)
        if multi_step_mode:
            input_spike_train = input.clone()
            output_spike_train = output.clone()
        else:
            input_spike_train = input[None]
            output_spike_train = output[None]
        delta_weight = torch.zeros_like(weight)
        if training:
            T, B, C, HI, WI = input_spike_train.shape
            T, B, N, HO, WO = output_spike_train.shape
            N, C, HK, WK = weight.shape
            SH, SW = stride
            PH, PW = padding
            DH, DW = dilation
            for y in range(HO):
                for x in range(WO):
                    for p in range(HK):
                        for q in range(WK):
                            u = y * SH + p * DH - PH
                            v = x * SW + q * DW - PW
                            if u < 0 or u >= HI or v < 0 or v >= WI:
                                continue
                            for t in range(T):
                                delta_weight[:, :, p, q], input_trace[:, :, :, u, v], output_trace[:, :, :, y, x] = _stdp_online(
                                    delta_weight = delta_weight[:, :, p, q], # [CO, CI]
                                    input_trace = input_trace[:, :, :, u, v], # [B, CO, CI]
                                    output_trace = output_trace[:, :, :, y, x], # [B, CO, CI]
                                    input_spike_train = input_spike_train[t, :, :, u, v], # [B, CI]
                                    output_spike_train = output_spike_train[t, :, :, y, x], # [B, CO]
                                    a_pos = a_pos,
                                    tau_pos = tau_pos,
                                    a_neg = a_neg,
                                    tau_neg = tau_neg
                                )
        ctx.save_for_backward(delta_weight, input)
        return output, input_trace, output_trace


    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor, grad_input_trace: torch.Tensor, grad_output_trace: torch.Tensor) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None, None, None, None, None]:
        """
        利用STDP进行学习的2维卷积层的反向传播函数。
        Args:
            ctx (Any): 上下文
            grad_output (torch.Tensor): 输出脉冲序列梯度
            grad_input_trace (torch.Tensor): 输入的迹梯度
            grad_output_trace (torch.Tensor): 输出的迹梯度
        Returns:
            grad_input (torch.Tensor): 输入脉冲序列梯度
            grad_weight (torch.Tensor): 权重矩阵梯度
            grad_input_trace (torch.Tensor): 输入的迹梯度
            grad_output_trace (torch.Tensor): 输出的迹梯度
            grad_soma (None): 胞体的梯度，为None
            grad_stride (size_t): 输出步长的梯度，为None
            grad_padding (size_t): 边缘填充的量的梯度，为None
            grad_dilation (size_t): 输入步长的梯度，为None
            grad_a_pos (None): STDP参数A+的梯度，为None
            grad_tau_pos (None): STDP参数tau+的梯度，为None
            grad_a_neg (None): STDP参数A-的梯度，为None
            grad_tau_neg (None): STDP参数tau-的梯度，为None
            grad_training (None): 是否正在训练的梯度，为None
            grad_multi_step_mode (None): 是否为多时间步模式的梯度，为None
        """
        delta_weight, input = ctx.saved_tensors
        delta_weight = -delta_weight
        return torch.zeros_like(input), delta_weight, torch.zeros_like(grad_input_trace), torch.zeros_like(grad_output_trace), None, None, None, None, None, None, None, None, None, None


class STDPConv2d(STDPLayer):
    def __init__(self, soma: _Soma, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, a_pos: float = 0.0002, tau_pos: float = 2.0, a_neg: float = 0.0002, tau_neg: float = 2.0, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        使用STDP学习机制时的2维卷积层。
        Args:
            soma (nn.Module): 使用的脉冲神经元胞体，在matterhorn_pytorch.snn.soma中选择
            in_channels (int): 输入的频道数C_{in}
            out_channels (int): 输出的频道C_{out}
            kernel_size (size_2_t): 卷积核的形状
            stride (size_2_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_2_t): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_2_t): 卷积的输入步长
            a_pos (float): STDP参数A+
            tau_pos (float): STDP参数tau+
            a_neg (float): STDP参数A-
            tau_neg (float): STDP参数tau-
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__(
            a_pos = a_pos,
            tau_pos = tau_pos,
            a_neg = a_neg,
            tau_neg = tau_neg
        )
        def _fill(data: _size_any_t, l: int) -> torch.Tensor:
            res = torch.tensor(data)
            if res.ndim == 0:
                res = res[None]
            if res.shape[0] < l:
                res = torch.cat([res] * l)
                res = res[:l]
            return tuple(res)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _fill(kernel_size, 2)
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]), device = device, dtype = dtype))
        nn.init.kaiming_uniform_(self.weight, a = 5.0 ** 0.5)
        self.stride = _fill(stride, 2)
        self.padding = _fill(padding, 2)
        self.dilation = _fill(dilation, 2)
        self.soma = soma


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["in_channels=%d" % self.in_channels, "out_channels=%d" % self.out_channels, "kernel_size=(%d, %d)" % tuple(self.kernel_size), "stride=(%d, %d)" % tuple(self.stride), "padding=(%d, %d)" % tuple(self.padding), "dilation=(%d, %d)" % tuple(self.dilation)]) + (", " + super().extra_repr(self) if len(super().extra_repr(self)) else "")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        if self.multi_step_mode:
            T, B, C, HI, WI = x.shape
        else:
            T = 1
            B, C, HI, WI = x.shape
        N, C, HK, WK = self.weight.shape
        PH, PW = self.padding
        SH, SW = self.stride
        DH, DW = self.dilation
        HO = (HI + 2 * PH - HK * DH) // SH + 1
        WO = (WI + 2 * PW - WK * DW) // SW + 1
        input_trace_shape = torch.zeros(B, N, C, HI, WI).to(x)
        output_trace_shape = torch.zeros(B, N, C, HO, WO).to(x)
        self.check_if_reset(self.input_trace, input_trace_shape)
        self.check_if_reset(self.output_trace, output_trace_shape)
        self.input_trace = _SF.to(self.input_trace, input_trace_shape)
        self.output_trace = _SF.to(self.output_trace, output_trace_shape)
        soma = self.soma.forward_steps if self.multi_step_mode else self.soma.forward_step
        y, self.input_trace, self.output_trace = f_stdp_conv2d.apply(x, self.weight, self.input_trace, self.output_trace, soma, self.stride, self.padding, self.dilation, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg, self.training, self.multi_step_mode)
        return y


class MaxPool1d(Layer, nn.MaxPool1d):
    def __init__(self, kernel_size: _size_any_t, stride: _Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, spike_mode: str = "s") -> None:
        """
        一维最大池化。
        Args:
            kernel_size (size_any_t): 池化核大小
            stride (size_any_t | None): 池化步长
            padding (size_any_t): 边界填充的长度
            dilation (size_any_t): 输入侧的池化步长
            return_indices (bool): 是否返回带索引的内容
            ceil_mode (bool): 是否向上取整
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.MaxPool1d.__init__(
            self,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            return_indices = return_indices,
            ceil_mode = ceil_mode
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.MaxPool1d.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.MaxPool1d.forward(self, x)
        return y


class MaxPool2d(Layer, nn.MaxPool2d):
    def __init__(self, kernel_size: _size_any_t, stride: _Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, spike_mode: str = "s") -> None:
        """
        二维最大池化。
        Args:
            kernel_size (size_any_t): 池化核大小
            stride (size_any_t | None): 池化步长
            padding (size_any_t): 边界填充的长度
            dilation (size_any_t): 输入侧的池化步长
            return_indices (bool): 是否返回带索引的内容
            ceil_mode (bool): 是否向上取整
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.MaxPool2d.__init__(
            self,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            return_indices = return_indices,
            ceil_mode = ceil_mode
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.MaxPool2d.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.MaxPool2d.forward(self, x)
        return y


class MaxPool3d(Layer, nn.MaxPool3d):
    def __init__(self, kernel_size: _size_any_t, stride: _Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, spike_mode: str = "s") -> None:
        """
        三维最大池化。
        Args:
            kernel_size (size_any_t): 池化核大小
            stride (size_any_t | None): 池化步长
            padding (size_any_t): 边界填充的长度
            dilation (size_any_t): 输入侧的池化步长
            return_indices (bool): 是否返回带索引的内容
            ceil_mode (bool): 是否向上取整
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.MaxPool3d.__init__(
            self,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            return_indices = return_indices,
            ceil_mode = ceil_mode
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.MaxPool3d.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.MaxPool3d.forward(self, x)
        return y


class AvgPool1d(Layer, nn.AvgPool1d):
    def __init__(self, kernel_size: _size_1_t, stride: _Optional[_size_1_t] = None, padding: _size_1_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, spike_mode: str = "s") -> None:
        """
        一维平均池化。
        Args:
            kernel_size (size_1_t): 池化核大小
            stride (size_1_t): 池化核步长
            padding (size_1_t): 边界填充的长度
            ceil_mode (bool): 是否向上取整
            count_include_pad (bool): 是否连带边界一起计算
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.AvgPool1d.__init__(
            self,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            ceil_mode = ceil_mode,
            count_include_pad = count_include_pad
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.AvgPool1d.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.AvgPool1d.forward(self, x)
        return y


class AvgPool2d(Layer, nn.AvgPool2d):
    def __init__(self, kernel_size: _size_2_t, stride: _Optional[_size_2_t] = None, padding: _size_2_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: _Optional[int] = None, spike_mode: str = "s") -> None:
        """
        二维平均池化。
        Args:
            kernel_size (size_2_t): 池化核大小
            stride (size_2_t | None): 池化核步长
            padding (size_2_t): 边界填充的长度
            ceil_mode (bool): 是否向上取整
            count_include_pad (bool): 是否连带边界一起计算
            divisor_override (int | None): 是否用某个数取代总和作为除数
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.AvgPool2d.__init__(
            self,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            ceil_mode = ceil_mode,
            count_include_pad = count_include_pad,
            divisor_override = divisor_override
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.AvgPool2d.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.AvgPool2d.forward(self, x)
        return y


class AvgPool3d(Layer, nn.AvgPool3d):
    def __init__(self, kernel_size: _size_3_t, stride: _Optional[_size_3_t] = None, padding: _size_3_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: _Optional[int] = None, spike_mode: str = "s") -> None:
        """
        三维平均池化。
        Args:
            kernel_size (size_3_t): 池化核大小
            stride (size_3_t | None): 池化核步长
            padding (size_3_t): 边界填充的长度
            ceil_mode (bool): 是否向上取整
            count_include_pad (bool): 是否连带边界一起计算
            divisor_override (int | None): 是否用某个数取代总和作为除数
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.AvgPool3d.__init__(
            self,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            ceil_mode = ceil_mode,
            count_include_pad = count_include_pad,
            divisor_override = divisor_override
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.AvgPool3d.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.AvgPool3d.forward(self, x)
        return y


class MaxUnpool1d(Layer, nn.MaxUnpool1d):
    def __init__(self, kernel_size: _Union[int, _Tuple[int]], stride: _Optional[_Union[int, _Tuple[int]]] = None, padding: _Union[int, _Tuple[int]] = 0, spike_mode: str = "s") -> None:
        """
        一维最大反池化。
        Args:
            kernel_size (size_3_t): 池化核大小
            stride (size_3_t | None): 池化核步长
            padding (size_3_t): 边界填充的长度
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.MaxUnpool1d.__init__(
            self,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.MaxUnpool1d.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")
    

    def forward_step(self, x: torch.Tensor, indices: torch.Tensor, output_size: _Optional[_Iterable[int]] = None) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
            indices (torch.Tensor): 池化前脉冲的索引
            output_size (torch.Tensor): 输出大小
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.MaxUnpool1d.forward(self, x, indices, output_size)
        return y


class MaxUnpool2d(Layer, nn.MaxUnpool2d):
    def __init__(self, kernel_size: _Union[int, _Tuple[int]], stride: _Optional[_Union[int, _Tuple[int]]] = None, padding: _Union[int, _Tuple[int]] = 0, spike_mode: str = "s") -> None:
        """
        二维最大反池化。
        Args:
            kernel_size (size_3_t): 池化核大小
            stride (size_3_t | None): 池化核步长
            padding (size_3_t): 边界填充的长度
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.MaxUnpool2d.__init__(
            self,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.MaxUnpool2d.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")
    

    def forward_step(self, x: torch.Tensor, indices: torch.Tensor, output_size: _Optional[_Iterable[int]] = None) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
            indices (torch.Tensor): 池化前脉冲的索引
            output_size (torch.Tensor): 输出大小
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.MaxUnpool2d.forward(self, x, indices, output_size)
        return y


class MaxUnpool3d(Layer, nn.MaxUnpool3d):
    def __init__(self, kernel_size: _Union[int, _Tuple[int]], stride: _Optional[_Union[int, _Tuple[int]]] = None, padding: _Union[int, _Tuple[int]] = 0, spike_mode: str = "s") -> None:
        """
        一维最大反池化。
        Args:
            kernel_size (size_3_t): 池化核大小
            stride (size_3_t | None): 池化核步长
            padding (size_3_t): 边界填充的长度
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.MaxUnpool3d.__init__(
            self,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.MaxUnpool3d.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor, indices: torch.Tensor, output_size: _Optional[_Iterable[int]] = None) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
            indices (torch.Tensor): 池化前脉冲的索引
            output_size (torch.Tensor): 输出大小
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.MaxUnpool3d.forward(self, x, indices, output_size)
        return y


class Upsample(Layer, nn.Upsample):
    def __init__(self, size: _Optional[_Union[int, _Tuple[int]]] = None, scale_factor: _Optional[_Union[float, _Tuple[float]]] = None, mode: str = 'nearest', align_corners: _Optional[bool] = None, recompute_scale_factor: _Optional[bool] = None, spike_mode: str = "s") -> None:
        """
        上采样（反池化）。
        Args:
            size (int | int*): 输出大小
            scale_factor (float | float*): 比例因子，如2为上采样两倍
            mode (str): 以何种形式上采样
            align_corners (bool): 若为True，使输入和输出张量的角像素对齐，从而保留这些像素的值
            recompute_scale_factor (bool): 若为True，则必须传入scale_factor并且scale_factor用于计算输出大小。计算出的输出大小将用于推断插值的新比例；若为False，那么size或scale_factor将直接用于插值
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.Upsample.__init__(
            self,
            size = size,
            scale_factor = scale_factor,
            mode = mode,
            align_corners = align_corners,
            recompute_scale_factor = recompute_scale_factor
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.Upsample.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Upsample.forward(self, x)
        return y


class Flatten(Layer, nn.Flatten):
    def __init__(self, start_dim: int = 1, end_dim: int = -1, spike_mode: str = "s") -> None:
        """
        展平层。
        Args:
            start_dim (int): 起始维度，默认为1
            end_dim (int): 终止维度
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.Flatten.__init__(
            self,
            start_dim = start_dim,
            end_dim = end_dim
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.Flatten.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Flatten.forward(self, x)
        return y


    def forward_steps(self, x: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        dim_frac = lambda x: x + 1 if x >= 0 else x
        y = torch.flatten(x, dim_frac(self.start_dim), dim_frac(self.end_dim))
        return y


class Unflatten(Layer, nn.Unflatten):
    def __init__(self, dim: _Union[int, str], unflattened_size: _size, spike_mode: str = "s") -> None:
        """
        反展开层。
        Args:
            dim (int | str): 在哪个维度反展开
            unflattened_size: 这个维度上的张量要反展开成什么形状
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.Unflatten.__init__(
            self,
            dim = dim,
            unflattened_size = unflattened_size
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.Unflatten.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Unflatten.forward(self, x)
        return y


    def forward_steps(self, x: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        dim_frac = lambda x: x + 1 if x >= 0 else x
        y = torch.unflatten(x, dim_frac(self.dim), self.unflattened_size)
        return y


class Dropout(Layer, nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False, spike_mode: str = "s") -> None:
        """
        遗忘层。
        Args:
            p (float): 遗忘概率
            inplace (bool): 是否在原有张量上改动，若为True则直接改原张量，否则新建一个张量
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.Dropout.__init__(
            self,
            p = p,
            inplace = inplace
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.Dropout.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Dropout.forward(self, x)
        return y


class Dropout1d(Layer, nn.Dropout1d):
    def __init__(self, p: float = 0.5, inplace: bool = False, spike_mode: str = "s") -> None:
        """
        一维遗忘层。
        Args:
            p (float): 遗忘概率
            inplace (bool): 是否在原有张量上改动，若为True则直接改原张量，否则新建一个张量
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.Dropout1d.__init__(
            self,
            p = p,
            inplace = inplace
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.Dropout1d.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Dropout1d.forward(self, x)
        return y


class Dropout2d(Layer, nn.Dropout2d):
    def __init__(self, p: float = 0.5, inplace: bool = False, spike_mode: str = "s") -> None:
        """
        二维遗忘层。
        Args:
            p (float): 遗忘概率
            inplace (bool): 是否在原有张量上改动，若为True则直接改原张量，否则新建一个张量
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.Dropout2d.__init__(
            self,
            p = p,
            inplace = inplace
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.Dropout2d.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Dropout2d.forward(self, x)
        return y


class Dropout3d(Layer, nn.Dropout3d):
    def __init__(self, p: float = 0.5, inplace: bool = False, spike_mode: str = "s") -> None:
        """
        三维遗忘层。
        Args:
            p (float): 遗忘概率
            inplace (bool): 是否在原有张量上改动，若为True则直接改原张量，否则新建一个张量
            spike_mode (str): 脉冲发放模式，有单值（"s"），多值（"m"）和不进行脉冲转换（None）3种
        """
        Layer.__init__(
            self,
            spike_mode = spike_mode
        )
        nn.Dropout3d.__init__(
            self,
            p = p,
            inplace = inplace
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.Dropout3d.extra_repr(self) + (", " + Layer.extra_repr(self) if len(Layer.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Dropout3d.forward(self, x)
        return y