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
from matterhorn_pytorch.snn.container import Temporal as _Temporal
from typing import Any as _Any, Tuple as _Tuple, Iterable as _Iterable, Optional as _Optional, Union as _Union
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from torch.types import _size
from matterhorn_pytorch.training.functional import stdp_online as _stdp_online


class Layer(_Module):
    def __init__(self, multi_time_step = False) -> None:
        """
        突触函数的骨架，定义突触最基本的函数。
        Args:
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        super().__init__(
            multi_time_step = multi_time_step
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return "multi_time_step=%s" % (str(self.multi_time_step),)


    def forward_multi_time_steps(self, *args, **kwargs) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            *args: 输入
            **kwargs: 输入
        Returns:
            res (torch.Tensor): 输出
        """
        args, kwargs, tb = _SF.merge_time_steps_batch_size(args, kwargs)
        res = self.forward_single_time_step(*args, **kwargs)
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
        if isinstance(res, _Tuple):
            res = (_SF.val_to_spike(y) if isinstance(y, torch.Tensor) else y for y in res)
        else:
            res = _SF.val_to_spike(res)
        return res


class f_stdp_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, input: torch.Tensor, weight: torch.Tensor, input_trace: torch.Tensor, output_trace: torch.Tensor, soma: _Module, a_pos: float = 0.015, tau_pos: float = 2.0, a_neg: float = 0.015, tau_neg: float = 2.0, training: bool = True, multi_time_step: bool = True) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            multi_time_step (bool): 是否为多时间步模式
        Returns:
            output (torch.Tensor): 输出脉冲序列
            input_trace (torch.Tensor): 输入的迹，累积的输入效应
            output_trace (torch.Tensor): 输出的迹，累积的输出效应
        """
        if multi_time_step:
            input_spike_train = input.clone()
            flattened_input, _, tb = _SF.merge_time_steps_batch_size(input)
            time_steps = tb[0]
            psp: torch.Tensor = _F.linear(flattened_input, weight, bias = None)
            psp = _SF.split_time_steps_batch_size(psp, tb)
        else:
            input_spike_train = input[None]
            time_steps = 1
            psp: torch.Tensor = _F.linear(input, weight, bias = None)
        output: torch.Tensor = soma(psp)
        if multi_time_step:
            output_spike_train = output.clone()
        else:
            output_spike_train = output[None]
        delta_weight = torch.zeros_like(weight)
        if training:
            for t in range(time_steps):
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
            grad_multi_time_step (None): 是否为多时间步模式的梯度，为None
        """
        delta_weight, input = ctx.saved_tensors
        delta_weight = -delta_weight
        return torch.zeros_like(input), delta_weight, torch.zeros_like(grad_input_trace), torch.zeros_like(grad_output_trace), None, None, None, None, None, None, None


class STDPLinear(_Module):
    def __init__(self, soma: _Module, in_features: int, out_features: int, a_pos: float = 0.015, tau_pos: float = 2.0, a_neg: float = 0.015, tau_neg: float = 2.0, multi_time_step: bool = True, reset_after_process: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
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
            multi_time_step (bool): 是否调整为多个时间步模式
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        self.input_trace = None
        self.output_trace = None
        super().__init__(
            multi_time_step = multi_time_step,
            reset_after_process = reset_after_process
        )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device = device, dtype = dtype))
        nn.init.kaiming_uniform_(self.weight, a = 5.0 ** 0.5)
        if self.multi_time_step:
            if soma.supports_multi_time_step:
                self.soma = soma.multi_time_step_(True)
            elif not soma.multi_time_step:
                self.soma = _Temporal(soma, reset_after_process = False)
        else:
            if soma.supports_single_time_step:
                self.soma = soma.multi_time_step_(False)
            else:
                self.soma = soma
        self.a_pos = a_pos
        self.tau_pos = tau_pos
        self.a_neg = a_neg
        self.tau_neg = tau_neg
        self.reset()


    def reset(self) -> nn.Module:
        """
        重置模型。
        """
        self.input_trace = _SF.reset_tensor(self.input_trace, 0.0)
        self.output_trace = _SF.reset_tensor(self.output_trace, 0.0)
        return super().reset()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        if self.multi_time_step:
            time_steps = x.shape[0]
            batch_size = x.shape[1]
        else:
            time_steps = 1
            batch_size = x.shape[0]
        trace_shape = torch.zeros_like(self.weight)[None].repeat_interleave(batch_size, dim = 0)
        self.input_trace = _SF.init_tensor(self.input_trace, trace_shape)
        self.output_trace = _SF.init_tensor(self.output_trace, trace_shape)
        y, self.input_trace, self.output_trace = f_stdp_linear.apply(x, self.weight, self.input_trace, self.output_trace, self.soma, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg, self.training, self.multi_time_step)
        return y


class f_stdp_conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, input: torch.Tensor, weight: torch.Tensor, input_trace: torch.Tensor, output_trace: torch.Tensor, soma: _Module, stride: _size_any_t, padding: _size_any_t, dilation: _size_any_t, a_pos: float = 0.015, tau_pos: float = 2.0, a_neg: float = 0.015, tau_neg: float = 2.0, training: bool = True, multi_time_step: bool = True) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            multi_time_step (bool): 是否为多时间步模式
        Returns:
            output (torch.Tensor): 输出脉冲序列
            input_trace (torch.Tensor): 输入的迹，累积的输入效应
            output_trace (torch.Tensor): 输出的迹，累积的输出效应
        """
        if multi_time_step:
            input_spike_train = input.clone()
            flattened_input, _, tb = _SF.merge_time_steps_batch_size(input)
            time_steps = tb[0]
            psp: torch.Tensor = _F.conv2d(flattened_input, weight, bias = None, stride = tuple(stride), padding = tuple(padding), dilation = tuple(dilation))
            psp = _SF.split_time_steps_batch_size(psp, tb)
        else:
            input_spike_train = input[None]
            time_steps = 1
            psp: torch.Tensor = _F.conv2d(input, weight, bias = None, stride = tuple(stride), padding = tuple(padding), dilation = tuple(dilation))
        output: torch.Tensor = soma(psp)
        if multi_time_step:
            output_spike_train = output.clone()
        else:
            output_spike_train = output[None]
        delta_weight = torch.zeros_like(weight)
        if training:
            h_in = input_spike_train.shape[3]
            w_in = input_spike_train.shape[4]
            h_out = output_spike_train.shape[3]
            w_out = output_spike_train.shape[4]
            h_wt = weight.shape[2]
            w_wt = weight.shape[3]
            h_stride = stride[0]
            w_stride = stride[1]
            h_padding = padding[0]
            w_padding = padding[1]
            h_dilation = dilation[0]
            w_dilation = dilation[1]
            for y in range(h_out):
                for x in range(w_out):
                    for p in range(h_wt):
                        for q in range(w_wt):
                            u = y * h_stride + p * h_dilation - h_padding
                            v = x * w_stride + q * w_dilation - w_padding
                            if u < 0 or u >= h_in or v < 0 or v >= w_in:
                                continue
                            for t in range(time_steps):
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
            grad_multi_time_step (None): 是否为多时间步模式的梯度，为None
        """
        delta_weight, input = ctx.saved_tensors
        delta_weight = -delta_weight
        return torch.zeros_like(input), delta_weight, torch.zeros_like(grad_input_trace), torch.zeros_like(grad_output_trace), None, None, None, None, None, None, None, None, None, None


class STDPConv2d(_Module):
    def __init__(self, soma: _Module, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, a_pos: float = 0.0002, tau_pos: float = 2.0, a_neg: float = 0.0002, tau_neg: float = 2.0, multi_time_step: bool = True, reset_after_process: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
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
            multi_time_step (bool): 是否调整为多个时间步模式
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        self.input_trace = None
        self.output_trace = None
        super().__init__(
            multi_time_step = multi_time_step,
            reset_after_process = reset_after_process
        )
        def _fill(data: _size_any_t, l: int) -> torch.Tensor:
            res = torch.tensor(data)
            if res.ndim == 0:
                res = res[None]
            if res.shape[0] < l:
                res = torch.cat([res] * l)
                res = res[:l]
            return res
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _fill(kernel_size, 2)
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]), device = device, dtype = dtype))
        nn.init.kaiming_uniform_(self.weight, a = 5.0 ** 0.5)
        self.stride = _fill(stride, 2)
        self.padding = _fill(padding, 2)
        self.dilation = _fill(dilation, 2)
        if self.multi_time_step:
            if soma.supports_multi_time_step:
                self.soma = soma.multi_time_step_(True)
            elif not soma.multi_time_step:
                self.soma = _Temporal(soma, reset_after_process = False)
        else:
            if soma.supports_single_time_step:
                self.soma = soma.multi_time_step_(False)
            else:
                self.soma = soma
        self.a_pos = a_pos
        self.tau_pos = tau_pos
        self.a_neg = a_neg
        self.tau_neg = tau_neg
        self.reset()


    def reset(self) -> nn.Module:
        """
        重置模型。
        """
        self.input_trace = _SF.reset_tensor(self.input_trace, 0.0)
        self.output_trace = _SF.reset_tensor(self.output_trace, 0.0)
        return super().reset()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        if self.multi_time_step:
            time_steps = x.shape[0]
            batch_size = x.shape[1]
            h_in = x.shape[3]
            w_in = x.shape[4]
        else:
            time_steps = 1
            batch_size = x.shape[0]
            h_in = x.shape[2]
            w_in = x.shape[3]
        c_out = self.weight.shape[0]
        c_in = self.weight.shape[1]
        h_wt = self.weight.shape[2]
        w_wt = self.weight.shape[3]
        h_out = (h_in + 2 * self.padding[0] - h_wt * self.dilation[0]) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - w_wt * self.dilation[1]) // self.stride[1] + 1
        self.input_trace = _SF.init_tensor(self.input_trace, torch.zeros(batch_size, c_out, c_in, h_in, w_in).to(x))
        self.output_trace = _SF.init_tensor(self.output_trace, torch.zeros(batch_size, c_out, c_in, h_out, w_out))
        y, self.input_trace, self.output_trace = f_stdp_conv2d.apply(x, self.weight, self.input_trace, self.output_trace, self.soma, self.stride, self.padding, self.dilation, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg, self.training, self.multi_time_step)
        return y


class MaxPool1d(Layer, nn.MaxPool1d):
    def __init__(self, kernel_size: _size_any_t, stride: _Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, multi_time_step: bool = False) -> None:
        """
        一维最大池化。
        Args:
            kernel_size (size_any_t): 池化核大小
            stride (size_any_t | None): 池化步长
            padding (size_any_t): 边界填充的长度
            dilation (size_any_t): 输入侧的池化步长
            return_indices (bool): 是否返回带索引的内容
            ceil_mode (bool): 是否向上取整
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
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
        return ", ".join([nn.MaxPool1d.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.MaxPool1d.forward(self, x)
        return y


class MaxPool2d(Layer, nn.MaxPool2d):
    def __init__(self, kernel_size: _size_any_t, stride: _Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, multi_time_step: bool = False) -> None:
        """
        二维最大池化。
        Args:
            kernel_size (size_any_t): 池化核大小
            stride (size_any_t | None): 池化步长
            padding (size_any_t): 边界填充的长度
            dilation (size_any_t): 输入侧的池化步长
            return_indices (bool): 是否返回带索引的内容
            ceil_mode (bool): 是否向上取整
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
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
        return ", ".join([nn.MaxPool2d.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.MaxPool2d.forward(self, x)
        return y


class MaxPool3d(Layer, nn.MaxPool3d):
    def __init__(self, kernel_size: _size_any_t, stride: _Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, multi_time_step: bool = False) -> None:
        """
        三维最大池化。
        Args:
            kernel_size (size_any_t): 池化核大小
            stride (size_any_t | None): 池化步长
            padding (size_any_t): 边界填充的长度
            dilation (size_any_t): 输入侧的池化步长
            return_indices (bool): 是否返回带索引的内容
            ceil_mode (bool): 是否向上取整
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
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
        return ", ".join([nn.MaxPool3d.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.MaxPool3d.forward(self, x)
        return y


class AvgPool1d(Layer, nn.AvgPool1d):
    def __init__(self, kernel_size: _size_1_t, stride: _Optional[_size_1_t] = None, padding: _size_1_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, multi_time_step: bool = False) -> None:
        """
        一维平均池化。
        Args:
            kernel_size (size_1_t): 池化核大小
            stride (size_1_t): 池化核步长
            padding (size_1_t): 边界填充的长度
            ceil_mode (bool): 是否向上取整
            count_include_pad (bool): 是否连带边界一起计算
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
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
        return ", ".join([nn.AvgPool1d.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.AvgPool1d.forward(self, x)
        return y


class AvgPool2d(Layer, nn.AvgPool2d):
    def __init__(self, kernel_size: _size_2_t, stride: _Optional[_size_2_t] = None, padding: _size_2_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: _Optional[int] = None, multi_time_step: bool = False) -> None:
        """
        二维平均池化。
        Args:
            kernel_size (size_2_t): 池化核大小
            stride (size_2_t | None): 池化核步长
            padding (size_2_t): 边界填充的长度
            ceil_mode (bool): 是否向上取整
            count_include_pad (bool): 是否连带边界一起计算
            divisor_override (int | None): 是否用某个数取代总和作为除数
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
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
        return ", ".join([nn.AvgPool2d.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.AvgPool2d.forward(self, x)
        return y


class AvgPool3d(Layer, nn.AvgPool3d):
    def __init__(self, kernel_size: _size_3_t, stride: _Optional[_size_3_t] = None, padding: _size_3_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: _Optional[int] = None, multi_time_step: bool = False) -> None:
        """
        三维平均池化。
        Args:
            kernel_size (size_3_t): 池化核大小
            stride (size_3_t | None): 池化核步长
            padding (size_3_t): 边界填充的长度
            ceil_mode (bool): 是否向上取整
            count_include_pad (bool): 是否连带边界一起计算
            divisor_override (int | None): 是否用某个数取代总和作为除数
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
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
        return ", ".join([nn.AvgPool3d.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.AvgPool3d.forward(self, x)
        return y


class MaxUnpool1d(Layer, nn.MaxUnpool1d):
    def __init__(self, kernel_size: _Union[int, _Tuple[int]], stride: _Optional[_Union[int, _Tuple[int]]] = None, padding: _Union[int, _Tuple[int]] = 0, multi_time_step: bool = False) -> None:
        """
        一维最大反池化。
        Args:
            kernel_size (size_3_t): 池化核大小
            stride (size_3_t | None): 池化核步长
            padding (size_3_t): 边界填充的长度
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
        )
        nn.MaxUnpool1d.__init__(
            self,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
    

    def forward_single_time_step(self, x: torch.Tensor, indices: torch.Tensor, output_size: _Optional[_Iterable[int]] = None) -> torch.Tensor:
        """
        前向传播函数。
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
    def __init__(self, kernel_size: _Union[int, _Tuple[int]], stride: _Optional[_Union[int, _Tuple[int]]] = None, padding: _Union[int, _Tuple[int]] = 0, multi_time_step: bool = False) -> None:
        """
        二维最大反池化。
        Args:
            kernel_size (size_3_t): 池化核大小
            stride (size_3_t | None): 池化核步长
            padding (size_3_t): 边界填充的长度
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
        )
        nn.MaxUnpool2d.__init__(
            self,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
    

    def forward_single_time_step(self, x: torch.Tensor, indices: torch.Tensor, output_size: _Optional[_Iterable[int]] = None) -> torch.Tensor:
        """
        前向传播函数。
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
    def __init__(self, kernel_size: _Union[int, _Tuple[int]], stride: _Optional[_Union[int, _Tuple[int]]] = None, padding: _Union[int, _Tuple[int]] = 0, multi_time_step: bool = False) -> None:
        """
        一维最大反池化。
        Args:
            kernel_size (size_3_t): 池化核大小
            stride (size_3_t | None): 池化核步长
            padding (size_3_t): 边界填充的长度
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
        )
        nn.MaxUnpool3d.__init__(
            self,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )


    def forward_single_time_step(self, x: torch.Tensor, indices: torch.Tensor, output_size: _Optional[_Iterable[int]] = None) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
            indices (torch.Tensor): 池化前脉冲的索引
            output_size (torch.Tensor): 输出大小
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.MaxUnpool3d.forward(self, x, indices, output_size)
        return y


class Upsample(nn.Upsample):
    def __init__(self, size: _Optional[_Union[int, _Tuple[int]]] = None, scale_factor: _Optional[_Union[float, _Tuple[float]]] = None, mode: str = 'nearest', align_corners: _Optional[bool] = None, recompute_scale_factor: _Optional[bool] = None) -> None:
        """
        上采样（反池化）。
        Args:
            size (int | int*): 输出大小
            scale_factor (float | float*): 比例因子，如2为上采样两倍
            mode (str): 以何种形式上采样
            align_corners (bool): 若为True，使输入和输出张量的角像素对齐，从而保留这些像素的值
            recompute_scale_factor (bool): 若为True，则必须传入scale_factor并且scale_factor用于计算输出大小。计算出的输出大小将用于推断插值的新比例；若为False，那么size或scale_factor将直接用于插值
        """
        nn.Upsample.__init__(
            self,
            size = size,
            scale_factor = scale_factor,
            mode = mode,
            align_corners = align_corners,
            recompute_scale_factor = recompute_scale_factor
        )


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Upsample.forward(self, x)
        return y


class Flatten(Layer, nn.Flatten):
    def __init__(self, start_dim: int = 2, end_dim: int = -1, multi_time_step: bool = False) -> None:
        """
        展平层。
        Args:
            start_dim (int): 起始维度，默认为2（除去[T, B]之后的维度）
            end_dim (int): 终止维度
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
        )
        nn.Flatten.__init__(
            self,
            start_dim = max(start_dim - 1, 0) if start_dim >= 0 else start_dim,
            end_dim = max(end_dim - 1, 0) if end_dim >= 0 else end_dim
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([nn.Flatten.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Flatten.forward(self, x)
        return y


class Unflatten(Layer, nn.Unflatten):
    def __init__(self, dim: _Union[int, str], unflattened_size: _size, multi_time_step: bool = False) -> None:
        """
        反展开层。
        Args:
            dim (int | str): 在哪个维度反展开
            unflattened_size: 这个维度上的张量要反展开成什么形状
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
        )
        nn.Unflatten.__init__(
            self,
            dim = max(dim - 1, 0) if dim >= 0 else dim,
            unflattened_size = unflattened_size
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([nn.Unflatten.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Unflatten.forward(self, x)
        return y


class Dropout(Layer, nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False, multi_time_step: bool = False) -> None:
        """
        遗忘层。
        Args:
            p (float): 遗忘概率
            inplace (bool): 是否在原有张量上改动，若为True则直接改原张量，否则新建一个张量
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
        )
        nn.Dropout.__init__(
            self,
            p = p,
            inplace = inplace
        )


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Dropout.forward(self, x)
        return y


class Dropout1d(Layer, nn.Dropout1d):
    def __init__(self, p: float = 0.5, inplace: bool = False, multi_time_step: bool = False) -> None:
        """
        一维遗忘层。
        Args:
            p (float): 遗忘概率
            inplace (bool): 是否在原有张量上改动，若为True则直接改原张量，否则新建一个张量
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
        )
        nn.Dropout1d.__init__(
            self,
            p = p,
            inplace = inplace
        )


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Dropout1d.forward(self, x)
        return y


class Dropout2d(Layer, nn.Dropout2d):
    def __init__(self, p: float = 0.5, inplace: bool = False, multi_time_step: bool = False) -> None:
        """
        二维遗忘层。
        Args:
            p (float): 遗忘概率
            inplace (bool): 是否在原有张量上改动，若为True则直接改原张量，否则新建一个张量
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
        )
        nn.Dropout2d.__init__(
            self,
            p = p,
            inplace = inplace
        )


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Dropout2d.forward(self, x)
        return y


class Dropout3d(Layer, nn.Dropout3d):
    def __init__(self, p: float = 0.5, inplace: bool = False, multi_time_step: bool = False) -> None:
        """
        三维遗忘层。
        Args:
            p (float): 遗忘概率
            inplace (bool): 是否在原有张量上改动，若为True则直接改原张量，否则新建一个张量
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
        )
        nn.Dropout3d.__init__(
            self,
            p = p,
            inplace = inplace
        )


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Dropout3d.forward(self, x)
        return y