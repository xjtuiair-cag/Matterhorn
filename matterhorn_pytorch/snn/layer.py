# -*- coding: UTF-8 -*-
"""
脉冲神经网络的整个神经元层，输入为脉冲，输出为脉冲。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
"""


from typing import Tuple, Callable, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from torch.types import _size
from matterhorn_pytorch.snn.container import Temporal
import matterhorn_pytorch.snn.functional as SF
from matterhorn_pytorch.snn.skeleton import Module
from matterhorn_pytorch.snn import surrogate
from matterhorn_pytorch.training.functional import stdp_online
try:
    from rich import print
except:
    pass


class Layer(Module):
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


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$，形状为[B, ...]
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$，形状为[B, ...]
        """
        y = x
        return y


    def forward_multi_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$，形状为[T, B, ...]
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$，形状为[T, B, ...]
        """
        time_steps = x.shape[0]
        batch_size = x.shape[1]
        x = x.flatten(0, 1)
        y = self.forward_single_time_step(x)
        output_shape = [time_steps, batch_size] + list(y.shape[1:])
        y = y.reshape(output_shape)
        return y


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        if self.multi_time_step:
            y = self.forward_multi_time_step(x)
        else:
            y = self.forward_single_time_step(x)
        y = SF.val_to_spike(y)
        return y


class f_stdp_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.Any, input: torch.Tensor, weight: torch.Tensor, input_trace: torch.Tensor, output_trace: torch.Tensor, soma: Module, a_pos: float = 0.25, tau_pos: float = 2.0, a_neg: float = 0.25, tau_neg: float = 2.0, training: bool = True, multi_time_step: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        利用STDP进行学习的液体状态机的前向传播函数。
        Args:
            ctx (torch.Any): 上下文
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
            time_steps = input.shape[0]
            batch_size = input.shape[1]
            flattened_input = input.flatten(0, 1)
            psp = F.linear(flattened_input, weight, bias = None)
            output_shape = [time_steps, batch_size] + list(psp.shape[1:])
            psp = psp.reshape(output_shape)
        else:
            input_spike_train = input[None]
            time_steps = 1
            batch_size = input.shape[0]
            psp = F.linear(input, weight, bias = None)
        output = soma(psp)
        if multi_time_step:
            output_spike_train = output.clone()
        else:
            output_spike_train = output[None]
        delta_weight = torch.zeros_like(weight)
        if training:
            for t in range(time_steps):
                delta_weight, input_trace, output_trace = stdp_online(
                    delta_weight = delta_weight,
                    input_trace = input_trace,
                    output_trace = output_trace,
                    input_spike_train = input_spike_train[t],
                    output_spike_train = output_spike_train[t],
                    a_pos = a_pos,
                    tau_pos = tau_pos,
                    a_neg = a_neg,
                    tau_neg = tau_neg
                )
        ctx.save_for_backward(delta_weight, input)
        return output, input_trace, output_trace


    @staticmethod
    def backward(ctx: torch.Any, grad_output: torch.Tensor, grad_input_trace: torch.Tensor, grad_output_trace: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None, None]:
        """
        利用STDP进行学习的液体状态机的反向传播函数。
        Args:
            ctx (torch.Any): 上下文
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


class STDPLinear(Module):
    def __init__(self, in_features: int, out_features: int, soma: Module, a_pos: float = 0.25, tau_pos: float = 2.0, a_neg: float = 0.25, tau_neg: float = 2.0, multi_time_step: bool = True, reset_after_process: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        使用STDP学习机制时的全连接层。
        Args:
            in_features (int): 输入长度，用法同nn.Linear
            out_features (int): 输出长度，用法同nn.Linear
            soma (nn.Module): 使用的脉冲神经元胞体，在matterhorn_pytorch.snn.soma中选择
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
            if soma.supports_multi_time_step():
                self.soma = soma.multi_time_step_(True)
            elif not soma.multi_time_step:
                self.soma = Temporal(soma, reset_after_process = False)
        else:
            if soma.supports_single_time_step():
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
        self.input_trace = SF.reset_tensor(self.input_trace, 0.0)
        self.output_trace = SF.reset_tensor(self.output_trace, 0.0)
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
        self.input_trace = SF.init_tensor(self.input_trace, trace_shape)
        self.output_trace = SF.init_tensor(self.output_trace, trace_shape)
        y, self.input_trace, self.output_trace = f_stdp_linear.apply(x, self.weight, self.input_trace, self.output_trace, self.soma, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg, self.training, self.multi_time_step)
        return y


class MaxPool1d(Layer, nn.MaxPool1d):
    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, multi_time_step: bool = False) -> None:
        """
        一维最大池化。
        Args:
            kernel_size (_size_any_t): 池化核大小
            stride (_size_any_t | None): 池化步长
            padding (_size_any_t): 边界填充的长度
            dilation (_size_any_t): 输入侧的池化步长
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
    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, multi_time_step: bool = False) -> None:
        """
        二维最大池化。
        Args:
            kernel_size (_size_any_t): 池化核大小
            stride (_size_any_t | None): 池化步长
            padding (_size_any_t): 边界填充的长度
            dilation (_size_any_t): 输入侧的池化步长
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
    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, multi_time_step: bool = False) -> None:
        """
        三维最大池化。
        Args:
            kernel_size (_size_any_t): 池化核大小
            stride (_size_any_t | None): 池化步长
            padding (_size_any_t): 边界填充的长度
            dilation (_size_any_t): 输入侧的池化步长
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
    def __init__(self, kernel_size: _size_1_t, stride: Optional[_size_1_t] = None, padding: _size_1_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, multi_time_step: bool = False) -> None:
        """
        一维平均池化。
        Args:
            kernel_size (_size_1_t): 池化核大小
            stride (_size_1_t): 池化核步长
            padding (_size_1_t): 边界填充的长度
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
    def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_2_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None, multi_time_step: bool = False) -> None:
        """
        二维平均池化。
        Args:
            kernel_size (_size_2_t): 池化核大小
            stride (_size_2_t | None): 池化核步长
            padding (_size_2_t): 边界填充的长度
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
    def __init__(self, kernel_size: _size_3_t, stride: Optional[_size_3_t] = None, padding: _size_3_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None, multi_time_step: bool = False) -> None:
        """
        三维平均池化。
        Args:
            kernel_size (_size_3_t): 池化核大小
            stride (_size_3_t | None): 池化核步长
            padding (_size_3_t): 边界填充的长度
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
    def __init__(self, dim: Union[int, str], unflattened_size: _size, multi_time_step: bool = False) -> None:
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