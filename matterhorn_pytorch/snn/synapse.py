# -*- coding: UTF-8 -*-
"""
脉冲神经网络神经元的突触，一层的前半段。输入为脉冲，输出为模拟电位值。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
"""


import torch
import torch.nn as nn
import torch.nn.functional as _F
import matterhorn_pytorch.snn.functional as _SF
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.normalization import _shape_t
from matterhorn_pytorch.snn.skeleton import Module as _Module
from typing import Any as _Any, Tuple as _Tuple, Iterable as _Iterable, Mapping as _Mapping, Union as _Union, Optional as _Optional


class Synapse(_Module):
    _required_ndims = None


    def __init__(self, batch_first: bool = False) -> None:
        """
        突触函数的骨架，定义突触最基本的函数。
        Args:
            batch_first (bool): 第一维为批(True)还是时间(False)
        """
        super().__init__()
        self.batch_first = batch_first


    def _check_ndim(self, x: torch.Tensor) -> None:
        if (self._required_ndims is not None) and (x.ndim not in self._required_ndims):
            raise AssertionError("Dimension of input tensor is required to be in %s, got %d." % (self._required_ndims, x.ndim))


    def _swap_batched_temporal(self, x: torch.Tensor, batched_ndim: int) -> torch.Tensor:
        if x.ndim < batched_ndim:
            return x
        if self.batch_first:
            x = x.swapaxes(0, 1) # [B, T] -> [T, B]
        return x


class _WeightStd(_Module):
    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-6, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        权重标准化。
        Args:
            num_features (int): 输出通道个数
            affine (bool): 是否对标准化后的权重进行仿射操作
            eps (float): 方差偏移量
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features, device = device, dtype = dtype), requires_grad = affine)
        self.beta = nn.Parameter(torch.zeros(num_features, device = device, dtype = dtype), requires_grad = False)


    @property
    def weight_std(self) -> torch.Tensor:
        assert hasattr(self, "weight"), "The property weight doesn't exist."
        weight: torch.Tensor = getattr(self, "weight")
        dims = tuple(range(1, weight.ndim))
        affine_shape = [self.num_features] + [1] * (weight.ndim - 1)
        n = torch.prod(torch.tensor(weight.shape[1:])).to(weight)
        mean = torch.mean(weight, dim = dims, keepdims = True).detach()
        var = torch.var(weight, dim = dims, keepdims = True).detach()
        weight = (weight - mean) / ((var * n) ** 0.5 + self.eps)
        gamma = self.gamma.reshape(*affine_shape)
        beta = self.beta.reshape(*affine_shape)
        weight = weight * gamma + beta
        return weight


class Linear(Synapse, nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        全连接操作，输入一个大小为[B, $L_{in}$]的张量，输出一个大小为[B, $L_{out}$]的张量。
        Args:
            in_features: 输入的长度$L_{in}$
            out_features: 输出的长度$L_{out}$
            bias (bool): 是否要加入偏置
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(
            self,
            batch_first = batch_first
        )
        nn.Linear.__init__(
            self,
            in_features = in_features,
            out_features = out_features,
            bias = bias,
            device = device,
            dtype = dtype
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([nn.Linear.extra_repr(self), Synapse.extra_repr(self)])

    
    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        o, shape = self._fold_for_parallel(o, max(2, o.ndim - 1))
        x = nn.Linear.forward(self, o)
        x = self._unfold_from_parallel(x, shape)
        return x


class WSLinear(_WeightStd, Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, affine: bool = True, eps: float = 1e-6, batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        权重标准化的全连接操作。
        Args:
            in_features: 输入的长度$L_{in}$
            out_features: 输出的长度$L_{out}$
            bias (bool): 是否要加入偏置
            affine (bool): 是否对标准化后的权重进行仿射操作
            eps (float): 方差偏移量
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Linear.__init__(
            self,
            in_features = in_features,
            out_features = out_features,
            bias = bias,
            device = device,
            dtype = dtype
        )
        _WeightStd.__init__(
            self,
            num_features = out_features,
            affine = affine,
            eps = eps,
            device = device,
            dtype = dtype
        )


    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        o, shape = self._fold_for_parallel(o, max(2, o.ndim - 1))
        x = _F.linear(o, self.weight_std, self.bias)
        x = self._unfold_from_parallel(x, shape)
        return x


class Conv1d(Synapse, nn.Conv1d):
    _required_ndims = (3, 4) # [T, C, L] / [B, T, C, L]


    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: _Union[_size_1_t, str] = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        一维卷积操作，输入一个大小为[B, $C_{in}$, $L_{in}$]的张量，输出一个大小为[B, $C_{out}$, $L_{out}$]的张量。
        Args:
            in_channels (int): 输入的频道数$C_{in}$
            out_channels (int): 输出的频道$C_{out}$
            kernel_size (size_1_t): 卷积核的形状
            stride (size_1_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_1_t | str): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_1_t): 卷积的输入步长
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            padding_mode (str): 边缘填充的方式
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(
            self,
            batch_first = batch_first
        )
        nn.Conv1d.__init__(
            self,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([nn.Conv1d.extra_repr(self), Synapse.extra_repr(self)])


    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        self._check_ndim(o)
        o, shape = self._fold_for_parallel(o, 3)
        x = nn.Conv1d.forward(self, o)
        x = self._unfold_from_parallel(x, shape)
        return x


class WSConv1d(_WeightStd, Conv1d):
    _required_ndims = (3, 4) # [T, C, L] / [B, T, C, L]


    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: _Union[_size_1_t, str] = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", affine: bool = True, eps: float = 1e-6, batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        权重标准化的一维卷积操作。
        Args:
            in_channels (int): 输入的频道数$C_{in}$
            out_channels (int): 输出的频道$C_{out}$
            kernel_size (size_1_t): 卷积核的形状
            stride (size_1_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_1_t | str): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_1_t): 卷积的输入步长
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            padding_mode (str): 边缘填充的方式
            affine (bool): 是否对标准化后的权重进行仿射操作
            eps (float): 方差偏移量
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Conv1d.__init__(
            self,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype
        )
        _WeightStd.__init__(
            self,
            num_features = out_channels,
            affine = affine,
            eps = eps,
            device = device,
            dtype = dtype
        )


    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        self._check_ndim(o)
        o, shape = self._fold_for_parallel(o, 3)
        x = _F.conv1d(o, self.weight_std, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x = self._unfold_from_parallel(x, shape)
        return x


class Conv2d(Synapse, nn.Conv2d):
    _required_ndims = (4, 5) # [T, C, H, W] / [B, T, C, H, W]


    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _Union[_size_2_t, str] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        二维卷积操作，输入一个大小为[B, $C_{in}$, $H_{in}$, $W_{in}$]的张量，输出一个大小为[B, $C_{out}$, $H_{out}$, $W_{out}$]的张量。
        Args:
            in_channels (int): 输入的频道数$C_{in}$
            out_channels (int): 输出的频道$C_{out}$
            kernel_size (size_2_t): 卷积核的形状
            stride (size_2_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_2_t | str): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_2_t): 卷积的输入步长
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            padding_mode (str): 边缘填充的方式
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(
            self,
            batch_first = batch_first
        )
        nn.Conv2d.__init__(
            self,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([nn.Conv2d.extra_repr(self), Synapse.extra_repr(self)])


    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        self._check_ndim(o)
        o, shape = self._fold_for_parallel(o, 4)
        x = nn.Conv2d.forward(self, o)
        x = self._unfold_from_parallel(x, shape)
        return x


class WSConv2d(_WeightStd, Conv2d):
    _required_ndims = (4, 5) # [T, C, H, W] / [B, T, C, H, W]


    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _Union[_size_2_t, str] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", affine: bool = True, eps: float = 1e-6, batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        权重标准化的二维卷积操作。
        Args:
            in_channels (int): 输入的频道数$C_{in}$
            out_channels (int): 输出的频道$C_{out}$
            kernel_size (size_2_t): 卷积核的形状
            stride (size_2_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_2_t | str): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_2_t): 卷积的输入步长
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            padding_mode (str): 边缘填充的方式
            affine (bool): 是否对标准化后的权重进行仿射操作
            eps (float): 方差偏移量
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Conv2d.__init__(
            self,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype
        )
        _WeightStd.__init__(
            self,
            num_features = out_channels,
            affine = affine,
            eps = eps,
            device = device,
            dtype = dtype
        )


    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        self._check_ndim(o)
        o, shape = self._fold_for_parallel(o, 4)
        x = _F.conv2d(o, self.weight_std, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x = self._unfold_from_parallel(x, shape)
        return x


class Conv3d(Synapse, nn.Conv3d):
    _required_ndims = (5, 6) # [T, C, L, H, W] / [B, T, C, L, H, W]


    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t = 1, padding: _Union[_size_3_t, str] = 0, dilation: _size_3_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        三维卷积操作，输入一个大小为[B, $C_{in}$, $H_{in}$, $W_{in}$, $L_{in}$]的张量，输出一个大小为[B, $C_{out}$, $H_{out}$, $W_{out}$, $L_{out}$]的张量。
        Args:
            in_channels (int): 输入的频道数$C_{in}$
            out_channels (int): 输出的频道$C_{out}$
            kernel_size (size_3_t): 卷积核的形状
            stride (size_3_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_3_t | str): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_3_t): 卷积的输入步长
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            padding_mode (str): 边缘填充的方式
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(
            self,
            batch_first = batch_first
        )
        nn.Conv3d.__init__(
            self,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([nn.Conv3d.extra_repr(self), Synapse.extra_repr(self)])


    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        self._check_ndim(o)
        o, shape = self._fold_for_parallel(o, 5)
        x = nn.Conv3d.forward(self, o)
        x = self._unfold_from_parallel(x, shape)
        return x


class WSConv3d(_WeightStd, Conv3d):
    _required_ndims = (5, 6) # [T, C, L, H, W] / [B, T, C, L, H, W]


    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t = 1, padding: _Union[_size_3_t, str] = 0, dilation: _size_3_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", affine: bool = True, eps: float = 1e-6, batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        权重标准化的三维卷积操作。
        Args:
            in_channels (int): 输入的频道数$C_{in}$
            out_channels (int): 输出的频道$C_{out}$
            kernel_size (size_3_t): 卷积核的形状
            stride (size_3_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_3_t | str): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_3_t): 卷积的输入步长
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            padding_mode (str): 边缘填充的方式
            affine (bool): 是否对标准化后的权重进行仿射操作
            eps (float): 方差偏移量
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Conv3d.__init__(
            self,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype
        )
        _WeightStd.__init__(
            self,
            num_features = out_channels,
            affine = affine,
            eps = eps,
            device = device,
            dtype = dtype
        )


    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        self._check_ndim(o)
        o, shape = self._fold_for_parallel(o, 5)
        x = _F.conv3d(o, self.weight_std, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x = self._unfold_from_parallel(x, shape)
        return x


class ConvTranspose1d(Synapse, nn.ConvTranspose1d):
    _required_ndims = (3, 4) # [T, C, L] / [B, T, C, L]


    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: _size_1_t = 0, output_padding: _size_1_t = 0, groups: int = 1, bias: bool = True, dilation: _size_1_t = 1, padding_mode: str = "zeros", batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        一维逆卷积操作，输入一个大小为[B, $C_{in}$, $L_{in}$]的张量，输出一个大小为[B, $C_{out}$, $L_{out}$]的张量。
        Args:
            in_channels (int): 输入的频道数$C_{in}$
            out_channels (int): 输出的频道$C_{out}$
            kernel_size (size_1_t): 卷积核的形状
            stride (size_1_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_1_t): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            output_padding (size_1_t): 在输出边缘填充的量
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            dilation (size_1_t): 卷积的输出步长
            padding_mode (str): 边缘填充的方式
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(
            self,
            batch_first = batch_first
        )
        nn.ConvTranspose1d.__init__(
            self,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            output_padding = output_padding,
            groups = groups,
            bias = bias,
            dilation = dilation,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([nn.ConvTranspose1d.extra_repr(self), Synapse.extra_repr(self)])


    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        self._check_ndim(o)
        o, shape = self._fold_for_parallel(o, 3)
        x = nn.ConvTranspose1d.forward(self, o)
        x = self._unfold_from_parallel(x, shape)
        return x


class ConvTranspose2d(Synapse, nn.ConvTranspose2d):
    _required_ndims = (4, 5) # [T, C, H, W] / [B, T, C, H, W]


    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t = 0, output_padding: _size_2_t = 0, groups: int = 1, bias: bool = True, dilation: _size_2_t = 1, padding_mode: str = "zeros", batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        二维逆卷积操作，输入一个大小为[B, $C_{in}$, $H_{in}$, $W_{in}$]的张量，输出一个大小为[B, $C_{out}$, $H_{out}$, $W_{out}$]的张量。
        Args:
            in_channels (int): 输入的频道数$C_{in}$
            out_channels (int): 输出的频道$C_{out}$
            kernel_size (size_2_t): 卷积核的形状
            stride (size_2_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_2_t): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            output_padding (size_2_t): 在输出边缘填充的量
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            dilation (size_2_t): 卷积的输出步长
            padding_mode (str): 边缘填充的方式
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(
            self,
            batch_first = batch_first
        )
        nn.ConvTranspose2d.__init__(
            self,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            output_padding = output_padding,
            groups = groups,
            bias = bias,
            dilation = dilation,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([nn.ConvTranspose2d.extra_repr(self), Synapse.extra_repr(self)])


    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        self._check_ndim(o)
        o, shape = self._fold_for_parallel(o, 4)
        x = nn.ConvTranspose2d.forward(self, o)
        x = self._unfold_from_parallel(x, shape)
        return x


class ConvTranspose3d(Synapse, nn.ConvTranspose3d):
    _required_ndims = (5, 6) # [T, C, L, H, W] / [B, T, C, L, H, W]


    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t = 1, padding: _size_3_t = 0, output_padding: _size_3_t = 0, groups: int = 1, bias: bool = True, dilation: _size_3_t = 1, padding_mode: str = "zeros", batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        三维逆卷积操作，输入一个大小为[B, $C_{in}$, $H_{in}$, $W_{in}$, $L_{in}$]的张量，输出一个大小为[B, $C_{out}$, $H_{out}$, $W_{out}$, $L_{out}$]的张量。
        Args:
            in_channels (int): 输入的频道数$C_{in}$
            out_channels (int): 输出的频道$C_{out}$
            kernel_size (size_3_t): 卷积核的形状
            stride (size_3_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_3_t): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            output_padding (size_3_t): 在输出边缘填充的量
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            dilation (size_3_t): 卷积的输出步长
            padding_mode (str): 边缘填充的方式
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(
            self,
            batch_first = batch_first
        )
        nn.ConvTranspose3d.__init__(
            self,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            output_padding = output_padding,
            groups = groups,
            bias = bias,
            dilation = dilation,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([nn.ConvTranspose3d.extra_repr(self), Synapse.extra_repr(self)])


    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        self._check_ndim(o)
        o, shape = self._fold_for_parallel(o, 5)
        x = nn.ConvTranspose3d.forward(self, o)
        x = self._unfold_from_parallel(x, shape)
        return x


class _BatchNorm(Synapse):
    def __init__(self, batch_first: bool = False) -> None:
        super().__init__(
            batch_first = batch_first
        )
    

    def _fold_for_parallel(self, x: torch.Tensor, target_dim: _Optional[int] = None) -> _Tuple[torch.Tensor, _Iterable[int]]:
        """
        将时间维度与特征维度后的那个维度合并，从而并行处理批归一化。
        Args:
            x (torch.Tensor): 未压缩前的张量
            target_dim (int): 目标维度
        Returns:
            x (torch.Tensor): 压缩后的张量
            shape (Tuple): 被压缩的形状信息
        """
        if not self.batch_first:
            x = x.swapaxes(0, 1) # [B, T, C, ...]
        x = x.swapaxes(1, 2) # [B, C, T, ...]
        shape = list(x.shape[2:4])
        x = x.flatten(2, 3) if x.ndim > 3 else x # [B, C, T * L]
        return x, shape
    

    def _unfold_from_parallel(self, x: torch.Tensor, shape: _Iterable[int]) -> torch.Tensor:
        """
        解压因并行而压缩的维度。
        Args:
            x (torch.Tensor): 压缩后的张量
            shape (Tuple): 被压缩的形状信息
        Returns:
            x (torch.Tensor): 未压缩前的张量
        """
        x = x.unflatten(2, shape) # [B, C, T] / [B, C, T, L]
        x = x.swapaxes(2, 1) # [B, T, C] / [B, T, C, L]
        if not self.batch_first:
            x = x.swapaxes(1, 0) # [T, B, C] / [T, B, C, L]
        return x


class BatchNorm1d(_BatchNorm, nn.BatchNorm1d):
    _required_ndims = (3, 4) # [B, T, C] / [B, T, C, L]


    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        一维批归一化。
        Args:
            num_features (int): 需要被归一化那一维度的长度
            eps (float): 参数epsilon
            momentum (float): 动量参数
            affine (bool): 是否启用参数gamma和beta，进行仿射变换
            track_running_stats (bool): 是否需要跟踪整个训练过程来进行批归一化的学习
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        _BatchNorm.__init__(
            self,
            batch_first = batch_first
        )
        nn.BatchNorm1d.__init__(
            self,
            num_features = num_features,
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.BatchNorm1d.extra_repr(self) + ((", " + Synapse.extra_repr(self)) if len(Synapse.extra_repr(self)) else "")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x)
        x = nn.BatchNorm1d.forward(self, x)
        x = self._unfold_from_parallel(x, shape)
        return x


class BatchNorm2d(_BatchNorm, nn.BatchNorm2d):
    _required_ndims = (5,) # [B, T, C, H, W]

    
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        二维批归一化。
        Args:
            num_features (int): 需要被归一化那一维度的长度
            eps (float): 参数epsilon
            momentum (float): 动量参数
            affine (bool): 是否启用参数gamma和beta，进行仿射变换
            track_running_stats (bool): 是否需要跟踪整个训练过程来进行批归一化的学习
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        _BatchNorm.__init__(
            self,
            batch_first = batch_first
        )
        nn.BatchNorm2d.__init__(
            self,
            num_features = num_features,
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([nn.BatchNorm2d.extra_repr(self), Synapse.extra_repr(self)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x)
        x = nn.BatchNorm2d.forward(self, x)
        x = self._unfold_from_parallel(x, shape)
        return x


class BatchNorm3d(_BatchNorm, nn.BatchNorm3d):
    _required_ndims = (6,) # [B, T, C, L, H, W]


    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        三维批归一化。
        Args:
            num_features (int): 需要被归一化那一维度的长度
            eps (float): 参数epsilon
            momentum (float): 动量参数
            affine (bool): 是否启用参数gamma和beta，进行仿射变换
            track_running_stats (bool): 是否需要跟踪整个训练过程来进行批归一化的学习
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        _BatchNorm.__init__(
            self,
            batch_first = batch_first
        )
        nn.BatchNorm3d.__init__(
            self,
            num_features = num_features,
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device = device,
            dtype = dtype
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([nn.BatchNorm3d.extra_repr(self), Synapse.extra_repr(self)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        self._check_ndim(x)
        x, shape = self._fold_for_parallel(x)
        x = nn.BatchNorm3d.forward(self, x)
        x = self._unfold_from_parallel(x, shape)
        return x


class LayerNorm(Synapse, nn.LayerNorm):
    def __init__(self, normalized_shape: _shape_t, eps: float = 0.00001, elementwise_affine: bool = True, batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        数据归一化。
        Args:
            normalized_shape (shape_t): 在什么数据尺度上进行归一化
            eps (float): 参数epsilon
            elementwise_affine (bool): 是否启用参数gamma和beta，进行仿射变换
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(
            self,
            batch_first = batch_first
        )
        nn.LayerNorm.__init__(
            self,
            normalized_shape = normalized_shape,
            eps = eps,
            elementwise_affine = elementwise_affine,
            device = device,
            dtype = dtype
        )


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join([nn.LayerNorm.extra_repr(self), Synapse.extra_repr(self)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x, shape = self._fold_for_parallel(x)
        x = nn.LayerNorm.forward(self, x)
        x = self._unfold_from_parallel(x, shape)
        return x


class MultiheadAttention(Synapse, nn.MultiheadAttention):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True, add_bias_kv: bool = False, add_zero_attn: bool = False, kdim: _Optional[int] = None, vdim: _Optional[int] = None, batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        多头自注意机制（默认序列长度为时间步）。
        Args:
            embed_dim (int): 特征维度$C$
            num_heads (int): 头的个数
            dropout (float): 遗忘率（丢弃信息的概率）
            bias (bool): 是否具有偏置
            add_bias_kv (bool): key和value是否具备特殊偏置
            add_zero_attn (bool): 是否在key和value上加一个新的批
            kdim (int | None): key的特征维度，默认为embed_dim
            vdim (int | None): value的特征维度，默认为embed_dim
            batch_first (bool): 第一维为批(True)还是时间(False)
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(
            self,
            batch_first = batch_first
        )
        nn.MultiheadAttention.__init__(
            self,
            embed_dim = embed_dim,
            num_heads = num_heads,
            dropout = dropout,
            bias = bias,
            add_bias_kv = add_bias_kv,
            add_zero_attn = add_zero_attn,
            kdim = kdim,
            vdim = vdim,
            batch_first = batch_first,
            device = device,
            dtype = dtype
        )
    

    def _fold_for_parallel(self, x: torch.Tensor, target_dim: int = 3) -> _Tuple[torch.Tensor, _Iterable[int]]:
        """
        将前几个维度压缩，以适应nn.Module中的预定义模块。
        Args:
            x (torch.Tensor): 未压缩前的张量
            target_dim (int): 目标维度
        Returns:
            x (torch.Tensor): 压缩后的张量
            shape (Tuple): 被压缩的形状信息
        """
        shape = []
        if x.ndim < target_dim:
            return x, shape
        if self.batch_first:
            x = x.swapaxes(0, 1) # [T, B, C, ...]
        x = x.swapaxes(1, 2) # [T, C, B, ...]
        if x.ndim > target_dim:
            shape = list(x.shape[target_dim - 1:])
            x = x.flatten(target_dim - 1) # [T, C, B']
        x = x.swapaxes(2, 1) # [T, B', C]
        if self.batch_first:
            x = x.swapaxes(1, 0) # [B', T, C]
        return x, shape


    def _unfold_from_parallel(self, x: torch.Tensor, shape: _Iterable[int]) -> torch.Tensor:
        """
        解压因并行而压缩的维度。
        Args:
            x (torch.Tensor): 压缩后的张量
            shape (Tuple): 被压缩的形状信息
        Returns:
            x (torch.Tensor): 未压缩前的张量
        """
        if x.ndim < 3:
            return x
        if self.batch_first:
            x = x.swapaxes(0, 1) # [T, B', C]
        x = x.swapaxes(1, 2) # [T, C, B']
        x = x.unflatten(x.ndim - 1, shape) # [T, C, B, ...]
        x = x.swapaxes(2, 1) # [T, B, C, ...]
        if self.batch_first:
            x = x.swapaxes(1, 0) # [B, T, C, ...]
        return x


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: _Optional[torch.Tensor] = None, need_weights: bool = True, attn_mask: _Optional[torch.Tensor] = None, average_attn_weights: bool = True) -> _Tuple[torch.Tensor, _Optional[torch.Tensor]]:
        """
        前向传播函数。
        Args:
            query (torch.Tensor): query张量$Q^{l}$，形状为[B, T, $C_{q}$](batch_first = True)或[T, B, $C_{q}$](batch_first = False)
            key (torch.Tensor): key张量$K^{l}$，形状为[B, $T_{kv}$, $C_{k}$](batch_first = True)或[$T_{kv}$, B, $C_{k}$](batch_first = False)
            value (torch.Tensor): value张量$V^{l}$，形状为[B, $T_{kv}$, $C_{v}$](batch_first = True)或[$T_{kv}$, B, $C_{v}$](batch_first = False)
            key_padding_mask (torch.Tensor | None): key的掩膜，指定哪些key不参与注意力计算，形状为[B, $T_{kv}$]
            need_weights (bool): 是否需要返回`attn_output_weights`
            attn_mask (torch.Tensor | None): 注意力掩膜，形状为[B * H, T, $T_{kv}$]
            average_attn_weights (bool): 若为True，返回的`attn_output_weights`会把所有头的做均值处理，否则单独返回每个头的`attn_output_weights`
        Returns:
            attn_output (torch.Tensor): 自注意力输出$Y^{l}$，形状为[B, T, C](batch_first = True)或[T, B, C](batch_first = False)
            attn_output_weights (torch.Tensor | None): 自注意力输出的权重，形状为[B, T, $T_{kv}$](average_attn_weights = True)或[B * H, T, $T_{kv}$](average_attn_weights = False)
        """
        query, q_shape = self._fold_for_parallel(query) # [L, B, C]
        key, k_shape = self._fold_for_parallel(key) # [L, B, C]
        value, v_shape = self._fold_for_parallel(value) # [L, B, C]
        l = lambda x: x.shape[0 if self.batch_first else 1]
        assert l(query) == l(value), "Shape of query (%d) not match shape of value (%d)." % (l(query), l(value))
        assert l(key) == l(value), "Shape of key (%d) not match shape of value (%d)." % (l(key), l(value))
        attn_output = nn.MultiheadAttention.forward(self, query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights)
        if need_weights:
            attn_output, attn_output_weights = attn_output
        else:
            attn_output_weights = None
        attn_output = self._unfold_from_parallel(attn_output, v_shape)
        w_shape = [el for el in v_shape]
        if attn_output_weights is not None and len(w_shape) > 1:
            if not average_attn_weights:
                w_shape[0] *= self.num_heads
            attn_output_weights = attn_output_weights.swapaxes(0, 1).swapaxes(1, 2) # [T, T_kv, B]
            attn_output_weights = attn_output_weights.unflatten(attn_output_weights.ndim - 1, w_shape) # [T, T_kv, B, ...]
            attn_output_weights = attn_output_weights.swapaxes(2, 1).swapaxes(1, 0) # [B, T, T_kv, ...]
        if need_weights:
            return attn_output, attn_output_weights
        else:
            return attn_output