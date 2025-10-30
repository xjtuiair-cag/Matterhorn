# -*- coding: UTF-8 -*-
"""
脉冲神经网络的整个神经元层，输入为脉冲，输出为脉冲。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
"""


import torch
import torch.nn as nn
import torch.nn.functional as _F
import matterhorn_pytorch.snn.functional as _SF
import matterhorn_pytorch.training.functional as _TF
from matterhorn_pytorch.snn.skeleton import Module as _Module
from matterhorn_pytorch.snn.soma import Soma as _Soma
from typing import Any as _Any, Tuple as _Tuple, Iterable as _Iterable, Mapping as _Mapping, Callable as _Callable, Optional as _Optional, Union as _Union
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from torch.types import _size


class Layer(_Module):
    _required_ndims = None


    def __init__(self, batch_first: bool = False) -> None:
        """
        层的骨架，定义层最基本的函数。
        Args:
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        super().__init__()
        self.batch_first = batch_first


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["batch_first=%s" % (self.batch_first)])


    def _check_ndim(self, x: torch.Tensor) -> None:
        if (self._required_ndims is not None) and (x.ndim not in self._required_ndims):
            raise AssertionError("Dimension of input tensor is required to be in %s, got %d." % (self._required_ndims, x.ndim))


    def _swap_batched_temporal(self, x: torch.Tensor, batched_ndim: int) -> torch.Tensor:
        if x.ndim < batched_ndim:
            return x
        if self.batch_first:
            x = x.swapaxes(0, 1) # [B, T] -> [T, B]
        return x


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
        return ", ".join(["pos=%g*exp(-x/%g)" % (self.a_pos, self.tau_pos), "neg=-%g*exp(x/%g)" % (self.a_neg, self.tau_neg)])


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


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([", ".join(["%d" % self.in_features, "%d" % self.out_features]), STDPLayer.extra_repr(self)])


    def forward(self, x: torch.Tensor, h_traces: _Optional[_Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
            h_traces (Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None): 当前层最初历史电位$H_{i}^{l}(0)$，输入迹和输出迹
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
            h_traces (Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None): 当前层最终历史电位$H_{i}^{l}(T)$，输入迹和输出迹
        """
        if h_traces is None:
            h_traces = (None, None, None)
        h, input_trace, output_trace = h_traces
        y, h, input_trace, output_trace = _TF.stdp_linear(x, self.weight, self.soma.forward, h, input_trace, output_trace, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg, self.training)
        return y, (h, input_trace, output_trace)


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
        return ", ".join([", ".join(["%d" % self.in_channels, "%d" % self.out_channels, "kernel_size=(%d, %d)" % tuple(self.kernel_size), "stride=(%d, %d)" % tuple(self.stride), "padding=(%d, %d)" % tuple(self.padding), "dilation=(%d, %d)" % tuple(self.dilation)]), STDPLayer.extra_repr(self)])


    def forward(self, x: torch.Tensor, h_traces: _Optional[_Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
            h_traces (Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None): 当前层最初历史电位$H_{i}^{l}(0)$，输入迹和输出迹
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
            h_traces (Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None): 当前层最终历史电位$H_{i}^{l}(T)$，输入迹和输出迹
        """
        if h_traces is None:
            h_traces = (None, None, None)
        h, input_trace, output_trace = h_traces
        y, h, input_trace, output_trace = _TF.stdp_conv2d(x, self.weight, self.soma.forward, self.stride, self.padding, self.dilation, h, input_trace, output_trace, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg, self.training)
        return y, (h, input_trace, output_trace)


class MaxPool1d(Layer, nn.MaxPool1d):
    _required_ndims = (3, 4) # [T, C, L] / [B, T, C, L]


    def __init__(self, kernel_size: _size_any_t, stride: _Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, batch_first: bool = False) -> None:
        """
        一维最大池化。
        Args:
            kernel_size (size_any_t): 池化核大小
            stride (size_any_t | None): 池化步长
            padding (size_any_t): 边界填充的长度
            dilation (size_any_t): 输入侧的池化步长
            return_indices (bool): 是否返回带索引的内容
            ceil_mode (bool): 是否向上取整
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        Layer.__init__(
            self,
            batch_first = batch_first
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x, 3)
        y = nn.MaxPool1d.forward(self, x)
        y = self._unfold_from_parallel(y, shape)
        return y


class MaxPool2d(Layer, nn.MaxPool2d):
    _required_ndims = (4, 5) # [T, C, H, W] / [B, T, C, H, W]


    def __init__(self, kernel_size: _size_any_t, stride: _Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, batch_first: bool = False) -> None:
        """
        二维最大池化。
        Args:
            kernel_size (size_any_t): 池化核大小
            stride (size_any_t | None): 池化步长
            padding (size_any_t): 边界填充的长度
            dilation (size_any_t): 输入侧的池化步长
            return_indices (bool): 是否返回带索引的内容
            ceil_mode (bool): 是否向上取整
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x, 4)
        y = nn.MaxPool2d.forward(self, x)
        y = self._unfold_from_parallel(y, shape)
        return y


class MaxPool3d(Layer, nn.MaxPool3d):
    _required_ndims = (5, 6) # [T, C, L, H, W] / [B, T, C, L, H, W]


    def __init__(self, kernel_size: _size_any_t, stride: _Optional[_size_any_t] = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False, batch_first: bool = False) -> None:
        """
        三维最大池化。
        Args:
            kernel_size (size_any_t): 池化核大小
            stride (size_any_t | None): 池化步长
            padding (size_any_t): 边界填充的长度
            dilation (size_any_t): 输入侧的池化步长
            return_indices (bool): 是否返回带索引的内容
            ceil_mode (bool): 是否向上取整
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x, 5)
        y = nn.MaxPool3d.forward(self, x)
        y = self._unfold_from_parallel(y, shape)
        return y


class AvgPool1d(Layer, nn.AvgPool1d):
    _required_ndims = (3, 4) # [T, C, L] / [B, T, C, L]


    def __init__(self, kernel_size: _size_1_t, stride: _Optional[_size_1_t] = None, padding: _size_1_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, batch_first: bool = False) -> None:
        """
        一维平均池化。
        Args:
            kernel_size (size_1_t): 池化核大小
            stride (size_1_t): 池化核步长
            padding (size_1_t): 边界填充的长度
            ceil_mode (bool): 是否向上取整
            count_include_pad (bool): 是否连带边界一起计算
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x, 3)
        y = nn.AvgPool1d.forward(self, x)
        y = self._unfold_from_parallel(y, shape)
        return y


class AvgPool2d(Layer, nn.AvgPool2d):
    _required_ndims = (4, 5) # [T, C, H, W] / [B, T, C, H, W]


    def __init__(self, kernel_size: _size_2_t, stride: _Optional[_size_2_t] = None, padding: _size_2_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: _Optional[int] = None, batch_first: bool = False) -> None:
        """
        二维平均池化。
        Args:
            kernel_size (size_2_t): 池化核大小
            stride (size_2_t | None): 池化核步长
            padding (size_2_t): 边界填充的长度
            ceil_mode (bool): 是否向上取整
            count_include_pad (bool): 是否连带边界一起计算
            divisor_override (int | None): 是否用某个数取代总和作为除数
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x, 4)
        y = nn.AvgPool2d.forward(self, x)
        y = self._unfold_from_parallel(y, shape)
        return y


class AvgPool3d(Layer, nn.AvgPool3d):
    _required_ndims = (5, 6) # [T, C, L, H, W] / [B, T, C, L, H, W]


    def __init__(self, kernel_size: _size_3_t, stride: _Optional[_size_3_t] = None, padding: _size_3_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: _Optional[int] = None, batch_first: bool = False) -> None:
        """
        三维平均池化。
        Args:
            kernel_size (size_3_t): 池化核大小
            stride (size_3_t | None): 池化核步长
            padding (size_3_t): 边界填充的长度
            ceil_mode (bool): 是否向上取整
            count_include_pad (bool): 是否连带边界一起计算
            divisor_override (int | None): 是否用某个数取代总和作为除数
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x, 5)
        y = nn.AvgPool3d.forward(self, x)
        y = self._unfold_from_parallel(y, shape)
        return y


class MaxUnpool1d(Layer, nn.MaxUnpool1d):
    _required_ndims = (3, 4) # [T, C, L] / [B, T, C, L]


    def __init__(self, kernel_size: _Union[int, _Tuple[int]], stride: _Optional[_Union[int, _Tuple[int]]] = None, padding: _Union[int, _Tuple[int]] = 0, batch_first: bool = False) -> None:
        """
        一维最大反池化。
        Args:
            kernel_size (size_3_t): 池化核大小
            stride (size_3_t | None): 池化核步长
            padding (size_3_t): 边界填充的长度
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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
        return ", ".join([nn.MaxUnpool1d.extra_repr(self), Layer.extra_repr(self)])
    

    def forward(self, x: torch.Tensor, indices: torch.Tensor, output_size: _Optional[_Iterable[int]] = None) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
            indices (torch.Tensor): 池化前脉冲的索引
            output_size (torch.Tensor): 输出大小
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x, 3)
        y = nn.MaxUnpool1d.forward(self, x, indices, output_size)
        y = self._unfold_from_parallel(y, shape)
        return y


class MaxUnpool2d(Layer, nn.MaxUnpool2d):
    _required_ndims = (4, 5) # [T, C, H, W] / [B, T, C, H, W]


    def __init__(self, kernel_size: _Union[int, _Tuple[int]], stride: _Optional[_Union[int, _Tuple[int]]] = None, padding: _Union[int, _Tuple[int]] = 0, batch_first: bool = False) -> None:
        """
        二维最大反池化。
        Args:
            kernel_size (size_3_t): 池化核大小
            stride (size_3_t | None): 池化核步长
            padding (size_3_t): 边界填充的长度
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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
        return ", ".join([nn.MaxUnpool2d.extra_repr(self), Layer.extra_repr(self)])
    

    def forward(self, x: torch.Tensor, indices: torch.Tensor, output_size: _Optional[_Iterable[int]] = None) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
            indices (torch.Tensor): 池化前脉冲的索引
            output_size (torch.Tensor): 输出大小
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x, 4)
        y = nn.MaxUnpool2d.forward(self, x, indices, output_size)
        y = self._unfold_from_parallel(y, shape)
        return y


class MaxUnpool3d(Layer, nn.MaxUnpool3d):
    _required_ndims = (5, 6) # [T, C, L, H, W] / [B, T, C, L, H, W]


    def __init__(self, kernel_size: _Union[int, _Tuple[int]], stride: _Optional[_Union[int, _Tuple[int]]] = None, padding: _Union[int, _Tuple[int]] = 0, batch_first: bool = False) -> None:
        """
        一维最大反池化。
        Args:
            kernel_size (size_3_t): 池化核大小
            stride (size_3_t | None): 池化核步长
            padding (size_3_t): 边界填充的长度
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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
        return ", ".join([nn.MaxUnpool3d.extra_repr(self), Layer.extra_repr(self)])


    def forward(self, x: torch.Tensor, indices: torch.Tensor, output_size: _Optional[_Iterable[int]] = None) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
            indices (torch.Tensor): 池化前脉冲的索引
            output_size (torch.Tensor): 输出大小
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x, 5)
        y = nn.MaxUnpool3d.forward(self, x, indices, output_size)
        y = self._unfold_from_parallel(y, shape)
        return y


class Upsample(Layer, nn.Upsample):
    def __init__(self, size: _Optional[_Union[int, _Tuple[int]]] = None, scale_factor: _Optional[_Union[float, _Tuple[float]]] = None, mode: str = 'nearest', align_corners: _Optional[bool] = None, recompute_scale_factor: _Optional[bool] = None, batch_first: bool = False) -> None:
        """
        上采样（反池化）。
        Args:
            size (int | int*): 输出大小
            scale_factor (float | float*): 比例因子，如2为上采样两倍
            mode (str): 以何种形式上采样
            align_corners (bool): 若为True，使输入和输出张量的角像素对齐，从而保留这些像素的值
            recompute_scale_factor (bool): 若为True，则必须传入scale_factor并且scale_factor用于计算输出大小。计算出的输出大小将用于推断插值的新比例；若为False，那么size或scale_factor将直接用于插值
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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
        return ", ".join([nn.Upsample.extra_repr(self), Layer.extra_repr(self)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        x, shape = self._fold_for_parallel(x)
        y = nn.Upsample.forward(self, x)
        y = self._unfold_from_parallel(y, shape)
        return y


class Flatten(Layer, nn.Flatten):
    def __init__(self, start_dim: int = 1, end_dim: int = -1, batch_first: bool = False) -> None:
        """
        展平层。
        Args:
            start_dim (int): 起始维度，默认为1
            end_dim (int): 终止维度
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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
        return ", ".join([nn.Flatten.extra_repr(self), Layer.extra_repr(self)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        x, shape = self._fold_for_parallel(x)
        y = nn.Flatten.forward(self, x)
        y = self._unfold_from_parallel(y, shape)
        return y


class Unflatten(Layer, nn.Unflatten):
    def __init__(self, dim: _Union[int, str], unflattened_size: _size, batch_first: bool = False) -> None:
        """
        反展开层。
        Args:
            dim (int | str): 在哪个维度反展开
            unflattened_size: 这个维度上的张量要反展开成什么形状
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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
        return ", ".join([nn.Unflatten.extra_repr(self), Layer.extra_repr(self)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        x, shape = self._fold_for_parallel(x)
        y = nn.Unflatten.forward(self, x)
        y = self._unfold_from_parallel(y, shape)
        return y


class Dropout(Layer, nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False, batch_first: bool = False) -> None:
        """
        遗忘层。
        Args:
            p (float): 遗忘概率
            inplace (bool): 是否在原有张量上改动，若为True则直接改原张量，否则新建一个张量
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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
        return ", ".join([nn.Dropout.extra_repr(self), Layer.extra_repr(self)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        x, shape = self._fold_for_parallel(x)
        y = nn.Dropout.forward(self, x)
        y = self._unfold_from_parallel(y, shape)
        return y


class Dropout1d(Layer, nn.Dropout1d):
    _required_ndims = (3, 4) # [T, C, L] / [B, T, C, L]


    def __init__(self, p: float = 0.5, inplace: bool = False, batch_first: bool = False) -> None:
        """
        一维遗忘层。
        Args:
            p (float): 遗忘概率
            inplace (bool): 是否在原有张量上改动，若为True则直接改原张量，否则新建一个张量
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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
        return ", ".join([nn.Dropout1d.extra_repr(self), Layer.extra_repr(self)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x, 3)
        y = nn.Dropout1d.forward(self, x)
        y = self._unfold_from_parallel(y, shape)
        return y


class Dropout2d(Layer, nn.Dropout2d):
    _required_ndims = (4, 5) # [T, C, H, W] / [B, T, C, H, W]


    def __init__(self, p: float = 0.5, inplace: bool = False, batch_first: bool = False) -> None:
        """
        二维遗忘层。
        Args:
            p (float): 遗忘概率
            inplace (bool): 是否在原有张量上改动，若为True则直接改原张量，否则新建一个张量
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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
        return ", ".join([nn.Dropout2d.extra_repr(self), Layer.extra_repr(self)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x, 4)
        y = nn.Dropout2d.forward(self, x)
        y = self._unfold_from_parallel(y, shape)
        return y


class Dropout3d(Layer, nn.Dropout3d):
    _required_ndims = (5, 6) # [T, C, L, H, W] / [B, T, C, L, H, W]


    def __init__(self, p: float = 0.5, inplace: bool = False, batch_first: bool = False) -> None:
        """
        三维遗忘层。
        Args:
            p (float): 遗忘概率
            inplace (bool): 是否在原有张量上改动，若为True则直接改原张量，否则新建一个张量
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        
        Layer.__init__(
            self,
            batch_first = batch_first
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
        return ", ".join([nn.Dropout3d.extra_repr(self), Layer.extra_repr(self)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 上一层脉冲$O_{j}^{l-1}(t)$
        Returns:
            y (torch.Tensor): 当前层脉冲$O_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x, 5)
        y = nn.Dropout3d.forward(self, x)
        y = self._unfold_from_parallel(y, shape)
        return y


class TemporalWiseAttention(Layer):
    def __init__(self, time_steps: int, d_threshold: float, expand: float = 1.0, batch_first: bool = False) -> None:
        """
        Tempora-wise Attention连接层：[Yao M, Gao H, Zhao G, et al. Temporal-wise attention spiking neural networks for event streams classification\[C\]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 10221-10230.](https://openaccess.thecvf.com/content/ICCV2021/html/Yao_Temporal-Wise_Attention_Spiking_Neural_Networks_for_Event_Streams_Classification_ICCV_2021_paper.html)
        Args:
            time_steps (int): 时间步长
            d_threshold (float): 注意阈值，用于阶跃函数
            expand (float): 权重矩阵的放缩率
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        super().__init__(
            batch_first = batch_first
        )
        mid_steps = int(time_steps * expand)
        self.attn = nn.Sequential(
            nn.Linear(time_steps, mid_steps, bias = False),
            nn.ReLU(),
            nn.Linear(mid_steps, time_steps, bias = False),
            nn.Sigmoid()
        )
        self.d_threshold = nn.Parameter(torch.tensor(d_threshold), requires_grad = False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        Args:
            x (torch.Tensor): 原始脉冲张量
        Returns:
            x (torch.Tensor): 过滤后的脉冲张量
        """
        dim = x.ndim
        s = x.clone().detach()
        pre_permute = []
        post_permute = []
        # 获取统计向量$s^{n-1}$
        if x.ndim >= 3:
            if self.batch_first:
                x = x.swapaxes(0, 1) # [T, B, ...]
            s = x.mean(dim = list(range(2, x.ndim))) # [T, B]
        else:
            s = x
        # 获取分数向量$d^{n-1}$：$d^{n-1}=TA(s^{n-1})$
        if s.ndim > 1:
            s = s.swapaxes(0, 1) # [B, T]
        d = self.attn(s) # [B, T]
        if not self.training:
            d = _SF.ge(d, self.d_threshold) # [B, T]
        if d.ndim > 1:
            d = d.swapaxes(0, 1) # [T, B]
        # 过滤：$X^{t,n-1}=d_{t}^{n-1}X^{t,n-1}$
        if x.ndim >= 3:
            x = x.swapaxes(1, -1).swapaxes(0, -2) # [..., T, B]
            x = d * x # [..., T, B]
            x = x.swapaxes(0, -2).swapaxes(1, -1) # [T, B, ...]
            if self.batch_first:
                x = x.swapaxes(0, 1) # [B, T, ...]
        else:
            x = d * x
        return x