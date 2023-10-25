# -*- coding: UTF-8 -*-
"""
脉冲神经网络的整个神经元层，输入为脉冲，输出为脉冲。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
"""


import math
from typing import Optional, Union
import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from torch.types import _size
from matterhorn_pytorch.snn.skeleton import Module
from matterhorn_pytorch.snn import surrogate
try:
    from rich import print
except:
    pass


class val_to_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        模拟值转脉冲的前向传播函数，以0.5为界
        @params:
            x: torch.Tensor 模拟值
        @return:
            o: torch.Tensor 脉冲值（0、1）
        """
        return x.ge(0.5).to(x)
    

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        模拟值转脉冲的反向传播函数
        @params:
            grad_output: torch.Tensor 输出梯度
        @return:
            grad_input: torch.Tensor 输入梯度
        """
        return grad_output


class SRM0Linear(Module):
    def __init__(self, in_features: int, out_features: int, tau_m: float = 2.0, u_threshold: float = 1.0, u_rest: float = 0.0, spiking_function: nn.Module = surrogate.Rectangular(), trainable: bool = False, device = None, dtype = None) -> None:
        """
        SRM0神经元，突触响应的神经元
        电位公式较为复杂：
        $$U_{i}(t)=η_{i}(t-t_{i})+\sum_{j}{w_{ij}\sum_{t_{j}^{(f)}}{ε_{ij}(t_{i}-t_{j}^{(f)})}}$$
        其中复位函数
        $$η_{i}(u)=-Θe^{-u^{m}+n}G(u)$$
        G(u)为矩形窗，当u∈[0,1)时为+∞，否则为1。
        突触响应函数
        $$ε_{ij}(s)=e^{-\frac{s}{τ_{m}}}H(s)$$
        H(s)为阶跃函数，当s>0时为1，否则为0。
        在此将其简化为多个突触反应与一个复位反应的叠加，即
        $$U_{i}^{l}(t)=u_{rest}+\sum_{j}{w_{ij}U_{ij}^{l}(t)}+R_{i}^{l}(t)$$
        @params:
            in_features: int 输入长度，用法同nn.Linear
            out_features: int 输出长度，用法同nn.Linear
            tau_m: float 膜时间常数$τ_{m}$
            u_threshold: float 阈电位$u_{th}$
            u_rest: float 静息电位$u_{rest}$
            spiking_function: nn.Module 计算脉冲时所使用的阶跃函数
            trainable: bool 参数是否可以训练
            device: Optional[torch.device, str] 所使用的设备
            dtype: Optional[type] 数据类型
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device = device, dtype = dtype), requires_grad = True)
        nn.init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        self.tau_m = nn.Parameter(torch.tensor(tau_m), requires_grad = trainable)
        self.u_threshold = u_threshold
        self.u_rest = u_rest
        self.spiking_function = spiking_function
        self.trainable = trainable
        self.reset()
    

    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return "in_features=%d, out_features=%d, tau_m=%g, u_threshold=%g, u_rest=%g, multi_time_step=%s, trainable=%s" % (self.in_features, self.out_features, self.tau_m, self.u_threshold, self.u_rest, str(self.multi_time_step), str(self.trainable))


    def reset(self) -> None:
        """
        重置整个神经元
        """
        self.s = 0.0
        self.r = 0.0


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
            u = u * torch.ones_like(x)
        return u


    def f_synapse_response(self, s: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """
        根据上一时刻的历史反应$S_{ij}^{l}(t-1)$与输入脉冲$O_{j}^{l-1}(t)$计算当前反应$S_{ij}^{l}(t)$。
        该部分可用如下公式概括：
        $$S_{ij}^{l}(t)=\frac{1}{\tau_{m}}S_{ij}^{l}(t-1)+O_{j}^{l-1}(t)$$
        @params:
            s: torch.Tensor 上一时刻的历史反应$S_{ij}^{l}(t-1)$
            o: torch.Tensor 输入脉冲$O_{j}^{l-1}(t)$
        @return:
            s: torch.Tensor 当前反应$S_{ij}^{l}(t)$
        """
        s = (1.0 / self.tau_m) * s + o
        return s


    def f_synapse_sum(self, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        根据当前反应$S_{ij}^{l}(t)$与权重$W_{ij}$求和计算当前电位$U_{i}^{l}(t)$。
        该部分可用如下公式概括：
        $$X_{i}^{l}(t)=\sum_{j}{w_{ij}S_{ij}^{l}(t)}$$
        @params:
            w: torch.Tensor 权重矩阵$W_{ij}$
            s: torch.Tensor 当前反应$S_{ij}^{l}(t)$
        @return:
            o: torch.Tensor 当前电位$U_{i}^{l}(t)$
        """
        u = nn.functional.linear(s, w)
        return u


    def f_response(self, r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的响应$R_{i}^{l}(t)$和当前时刻的输入电位$X_{i}^{l}(t)$计算当前电位$U_{i}^{l}(t)$。
        该部分可用如下公式概括：
        $$U_{i}^{l}(t)=X_{i}^{l}(t)*R_{i}^{l}(t)+u_{rest}$$
        @params:
            r: torch.Tensor 上一时刻的电位$U_{i}^{l}(t-1)$
            x: torch.Tensor 输入电位$X_{i}^{l}(t)$
        @return:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        """
        u = self.u_rest + (x * r)
        return u
    

    def f_firing(self, u: torch.Tensor) -> torch.Tensor:
        """
        通过当前电位$U_{i}^{l}(t)$计算当前脉冲$O_{i}^{l}(t)$。
        该部分可用如下公式概括：
        $$U_{i}^{l}(t)=X_{i}^{l}(t)+R_{i}^{l}(t)+u_{rest}$$
        $$O_{i}^{l}(t)=Heaviside(U_{i}^{l}(t)-u_{th})$$
        @params:
            u: torch.Tensor 当前电位$U_{i}^{l}(t)$
        @return:
            o: torch.Tensor 当前脉冲$O_{i}^{l}(t)$
        """
        return self.spiking_function(u - self.u_threshold)


    def f_reset(self, u: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """
        通过上一时刻的重置电位$R_{i}^{l}(t-1)$与当前脉冲$O_{i}^{l}(t-1)$得到当前重置电位$R_{i}^{l}(t)$。
        该部分可用如下公式概括：
        $$R_{i}^{l}(t)=\frac{1}{\tau_{r}}R_{i}^{l}(t-1)-m(u_{th}-u_{rest})O_{i}^{l}(t-1)$$
        此处将其改为是否产生不应期，得到：
        $$R_{i}^{l}(t)=-(u_{th}-u_{rest})O_{i}^{l}(t-1)$$
        @params:
            u: torch.Tensor 上一时刻的重置电位$R_{i}^{l}(t-1)$
            o: torch.Tensor 当前脉冲$O_{i}^{l}(t-1)$
        @return:
            r: torch.Tensor 当前重置电位$R_{i}^{l}(t)$
        """
        r = 1.0 - o
        return r


    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            o: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$
        @return:
            o: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$
        """
        # 突触函数
        # [batch_size, input_shape] -> [batch_size, output_shape]
        self.s = self.init_tensor(self.s, o)
        self.s = self.f_synapse_response(self.s, o)
        x = self.f_synapse_sum(self.weight, self.s)

        # 胞体函数，仍旧遵循R-S-R三段式
        # [batch_size, output_shape] -> [batch_size, output_shape]
        self.r = self.init_tensor(self.r, x)
        u = self.f_response(self.r, x)
        o = self.f_firing(u)
        self.r = self.f_reset(u, o)
        return o


class Layer(Module):
    def __init__(self, multi_time_step = False) -> None:
        """
        突触函数的骨架，定义突触最基本的函数。
        @params:
            multi_time_step: bool 是否调整为多个时间步模式
        """
        super().__init__(
            multi_time_step = multi_time_step
        )
        self.trainable = True


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return "multi_time_step=%s, trainable=%s" % (str(self.multi_time_step), str(self.trainable))


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        @params:
            x: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$，形状为[B, ...]
        @return:
            y: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$，形状为[B, ...]
        """
        y = x
        return y


    def forward_multi_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        @params:
            x: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$，形状为[T, B, ...]
        @return:
            y: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$，形状为[T, B, ...]
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
        @params:
            x: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$
        @return:
            y: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$
        """
        if self.multi_time_step:
            y = self.forward_multi_time_step(x)
        else:
            y = self.forward_single_time_step(x)
        y = val_to_spike.apply(y)
        return y


class MaxPool1d(Layer, nn.MaxPool1d):
    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, multi_time_step = False) -> None:
        """
        一维最大池化。
        @params:
            kernel_size: _size_any_t 池化核大小
            stride: _size_any_t | None 池化步长
            padding: _size_any_t 边界填充的长度
            dilation: _size_any_t 输入侧的池化步长
            return_indices: bool 是否返回带索引的内容
            ceil_mode: bool 是否向上取整
            multi_time_step: bool 是否调整为多个时间步模式
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
        @return:
            repr_str: str 参数表
        """
        return ", ".join([nn.MaxPool1d.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            x: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$
        @return:
            y: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.MaxPool1d.forward(self, x)
        return y


class MaxPool2d(Layer, nn.MaxPool2d):
    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, multi_time_step = False) -> None:
        """
        二维最大池化。
        @params:
            kernel_size: _size_any_t 池化核大小
            stride: _size_any_t | None 池化步长
            padding: _size_any_t 边界填充的长度
            dilation: _size_any_t 输入侧的池化步长
            return_indices: bool 是否返回带索引的内容
            ceil_mode: bool 是否向上取整
            multi_time_step: bool 是否调整为多个时间步模式
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
        @return:
            repr_str: str 参数表
        """
        return ", ".join([nn.MaxPool2d.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            x: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$
        @return:
            y: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.MaxPool2d.forward(self, x)
        return y


class MaxPool3d(Layer, nn.MaxPool3d):
    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, multi_time_step = False) -> None:
        """
        三维最大池化。
        @params:
            kernel_size: _size_any_t 池化核大小
            stride: _size_any_t | None 池化步长
            padding: _size_any_t 边界填充的长度
            dilation: _size_any_t 输入侧的池化步长
            return_indices: bool 是否返回带索引的内容
            ceil_mode: bool 是否向上取整
            multi_time_step: bool 是否调整为多个时间步模式
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
        @return:
            repr_str: str 参数表
        """
        return ", ".join([nn.MaxPool3d.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            x: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$
        @return:
            y: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.MaxPool3d.forward(self, x)
        return y


class AvgPool1d(Layer, nn.AvgPool1d):
    def __init__(self, kernel_size: _size_1_t, stride: _size_1_t = None, padding: _size_1_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, multi_time_step = False) -> None:
        """
        一维平均池化。
        @params:
            kernel_size: _size_1_t 池化核大小
            stride: _size_1_t 池化核步长
            padding: _size_1_t 边界填充的长度
            ceil_mode: bool 是否向上取整
            count_include_pad: bool 是否连带边界一起计算
            multi_time_step: bool 是否调整为多个时间步模式
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
        @return:
            repr_str: str 参数表
        """
        return ", ".join([nn.AvgPool1d.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            x: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$
        @return:
            y: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.AvgPool1d.forward(self, x)
        return y


class AvgPool2d(Layer, nn.AvgPool2d):
    def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_2_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None, multi_time_step = False) -> None:
        """
        二维平均池化。
        @params:
            kernel_size: _size_2_t 池化核大小
            stride: _size_2_t | None 池化核步长
            padding: _size_2_t 边界填充的长度
            ceil_mode: bool 是否向上取整
            count_include_pad: bool 是否连带边界一起计算
            divisor_override: int | None 是否用某个数取代总和作为除数
            multi_time_step: bool 是否调整为多个时间步模式
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
        @return:
            repr_str: str 参数表
        """
        return ", ".join([nn.AvgPool2d.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            x: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$
        @return:
            y: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.AvgPool2d.forward(self, x)
        return y


class AvgPool3d(Layer, nn.AvgPool3d):
    def __init__(self, kernel_size: _size_3_t, stride: Optional[_size_3_t] = None, padding: _size_3_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None, multi_time_step = False) -> None:
        """
        三维平均池化。
        @params:
            kernel_size: _size_3_t 池化核大小
            stride: _size_3_t | None 池化核步长
            padding: _size_3_t 边界填充的长度
            ceil_mode: bool 是否向上取整
            count_include_pad: bool 是否连带边界一起计算
            divisor_override: int | None 是否用某个数取代总和作为除数
            multi_time_step: bool 是否调整为多个时间步模式
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
        @return:
            repr_str: str 参数表
        """
        return ", ".join([nn.AvgPool3d.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            x: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$
        @return:
            y: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.AvgPool3d.forward(self, x)
        return y


class Flatten(Layer, nn.Flatten):
    def __init__(self, start_dim: int = 1, end_dim: int = -1, multi_time_step = False) -> None:
        """
        展平层。
        @params:
            start_dim: int 起始维度
            end_dim: int 终止维度
            multi_time_step: bool 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
        )
        nn.Flatten.__init__(
            self,
            start_dim = start_dim,
            end_dim = end_dim
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return ", ".join([nn.Flatten.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            x: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$
        @return:
            y: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Flatten.forward(self, x)
        return y


class Unflatten(Layer, nn.Unflatten):
    def __init__(self, dim: Union[int, str], unflattened_size: _size, multi_time_step = False) -> None:
        """
        反展开层。
        @params:
            dim: int | str 在哪个维度反展开
            unflattened_size: 这个维度上的张量要反展开成什么形状
            multi_time_step: bool 是否调整为多个时间步模式
        """
        Layer.__init__(
            self,
            multi_time_step = multi_time_step
        )
        nn.Unflatten.__init__(
            self,
            dim = dim,
            unflattened_size = unflattened_size
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return ", ".join([nn.Unflatten.extra_repr(self), Layer.extra_repr(self)])


    def forward_single_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        @params:
            x: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$
        @return:
            y: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$
        """
        y = nn.Unflatten.forward(self, x)
        return y