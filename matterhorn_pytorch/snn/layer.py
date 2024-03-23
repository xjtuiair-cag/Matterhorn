# -*- coding: UTF-8 -*-
"""
脉冲神经网络的整个神经元层，输入为脉冲，输出为脉冲。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
"""


import math
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from torch.types import _size
from matterhorn_pytorch.snn.container import Temporal
import matterhorn_pytorch.snn.functional as SF
from matterhorn_pytorch.snn.skeleton import Module
from matterhorn_pytorch.snn import surrogate
from matterhorn_pytorch.training.functional import stdp
try:
    from rich import print
except:
    pass


class STDPLinear(Module, nn.Linear):
    def __init__(self, in_features: int, out_features: int, soma: Module, a_pos: float = 0.05, tau_pos: float = 2.0, a_neg: float = 0.05, tau_neg: float = 2.0, lr: float = 0.01, multi_time_step: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        使用STDP学习机制时的全连接层
        Args:
            in_features (int): 输入长度，用法同nn.Linear
            out_features (int): 输出长度，用法同nn.Linear
            soma (nn.Module): 使用的脉冲神经元胞体，在matterhorn_pytorch.snn.soma中选择
            a_pos (float): STDP参数A+
            tau_pos (float): STDP参数tau+
            a_neg (float): STDP参数A-
            tau_neg (float): STDP参数tau-
            multi_time_step (bool): 是否调整为多个时间步模式
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Module.__init__(
            self,
            multi_time_step = multi_time_step,
            reset_after_process = False
        )
        nn.Linear.__init__(
            self,
            in_features = in_features, 
            out_features = out_features,
            bias = False,
            device = device,
            dtype = dtype
        )
        self.input_spike_seq = []
        self.output_spike_seq = []
        self.weight[:] = self.weight.requires_grad_(False)
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
        self.lr = lr
        self.reset()


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([nn.Linear.extra_repr(self), "a_pos=%g, tau_pos=%g, a_neg=%g, tau_neg=%g, lr=%g" % (self.a_pos, self.tau_pos, self.a_neg, self.tau_neg, self.lr)])


    def step(self, *args, **kwargs) -> Module:
        """
        对整个神经元应用STDP使其更新。
        """
        if not self.training:
            return
        if self.multi_time_step:
            input_spike_train = torch.cat(self.input_spike_seq)
            output_spike_train = torch.cat(self.output_spike_seq)
        else:
            input_spike_train = torch.stack(self.input_spike_seq)
            output_spike_train = torch.stack(self.output_spike_seq)
        is_batched = input_spike_train.ndim >= 3
        if is_batched:
            input_spike_train = input_spike_train.permute(1, 0, 2)
            output_spike_train = output_spike_train.permute(1, 0, 2)
        else:
            input_spike_train = input_spike_train.reshape(input_spike_train.shape[0], 1, input_spike_train.shape[1])
            output_spike_train = output_spike_train.reshape(output_spike_train.shape[0], 1, output_spike_train.shape[1])
        # 将不同维度的输入与输出张量形状统一转为[T, B, L]
        delta_weight = torch.zeros_like(self.weight)
        delta_weight = stdp(delta_weight, input_spike_train, output_spike_train, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg)
        self.weight += self.lr * delta_weight
        self.input_spike_seq = []
        self.output_spike_seq = []
        return super().step(*args, **kwargs)


    def forward_single_time_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。True
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$，形状为[B, I]
        Returns:
            o (torch.Tensor): 当前层的输出脉冲$O_{i}^{l}(t)$，形状为[B, O]
        """
        x = nn.Linear.forward(self, o)
        o = self.soma(x)
        return o


    def forward_multi_time_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}$，形状为[T, B, I]
        Returns:
            o (torch.Tensor): 当前层的输出脉冲$O_{i}^{l}$，形状为[T, B, O]
        """
        time_steps = o.shape[0]
        batch_size = o.shape[1]
        o = o.flatten(0, 1)
        x = nn.Linear.forward(self, o)
        output_shape = [time_steps, batch_size] + list(x.shape[1:])
        x = x.reshape(output_shape)
        o = self.soma(x)
        return o


    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            o (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        if self.training:
            self.input_spike_seq.append(o.clone().detach())
        if self.multi_time_step:
            o = self.forward_multi_time_step(o)
        else:
            o = self.forward_single_time_step(o)
        if self.training:
            self.output_spike_seq.append(o.clone().detach())
        return o


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
            if self.reset_after_process:
                self.reset()
        else:
            y = self.forward_single_time_step(x)
        y = SF.val_to_spike(y)
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