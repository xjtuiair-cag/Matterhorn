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
from typing import Any as _Any, Tuple as _Tuple, Mapping as _Mapping, Union as _Union, Optional as _Optional


class Synapse(_Module):
    def __init__(self) -> None:
        """
        突触函数的骨架，定义突触最基本的函数。
        """
        super().__init__()


    def forward_steps(self, *args: _Tuple[torch.Tensor], **kwargs: _Mapping[str, _Any]) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            *args (*torch.Tensor): 输入
            **kwargs (str: Any): 输入
        Returns:
            res (torch.Tensor): 输出
        """
        time_steps, batch_size = args[0].shape[:2]
        args = [_SF.merge_time_steps_batch_size(arg)[0] for arg in args]
        res = self.forward_step(*args, **kwargs)
        res = _SF.split_time_steps_batch_size(res, (time_steps, batch_size))
        return res


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
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        全连接操作，输入一个大小为[B, L_{in}]的张量，输出一个大小为[B, L_{out}]的张量。
        Args:
            in_features: 输入的长度L_{in}
            out_features: 输出的长度L_{out}
            bias (bool): 是否要加入偏置
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(self)
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
        return nn.Linear.extra_repr(self) + ((", " + Synapse.extra_repr(self)) if len(Synapse.extra_repr(self)) else "")

    
    def forward_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = nn.Linear.forward(self, o)
        return x


class WSLinear(_WeightStd, Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, affine: bool = True, eps: float = 1e-6, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        权重标准化的全连接操作。
        Args:
            in_features: 输入的长度L_{in}
            out_features: 输出的长度L_{out}
            bias (bool): 是否要加入偏置
            affine (bool): 是否对标准化后的权重进行仿射操作
            eps (float): 方差偏移量
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


    def forward_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = _F.linear(o, self.weight_std, self.bias)
        return x


class Conv1d(Synapse, nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: _Union[_size_1_t, str] = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        一维卷积操作，输入一个大小为[B, C_{in}, L_{in}]的张量，输出一个大小为[B, C_{out}, L_{out}]的张量。
        Args:
            in_channels (int): 输入的频道数C_{in}
            out_channels (int): 输出的频道C_{out}
            kernel_size (size_1_t): 卷积核的形状
            stride (size_1_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_1_t | str): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_1_t): 卷积的输入步长
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            padding_mode (str): 边缘填充的方式
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(self)
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
        return nn.Conv1d.extra_repr(self) + ((", " + Synapse.extra_repr(self)) if len(Synapse.extra_repr(self)) else "")


    def forward_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = nn.Conv1d.forward(self, o)
        return x


class WSConv1d(_WeightStd, Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: _Union[_size_1_t, str] = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", affine: bool = True, eps: float = 1e-6, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        权重标准化的一维卷积操作。
        Args:
            in_channels (int): 输入的频道数C_{in}
            out_channels (int): 输出的频道C_{out}
            kernel_size (size_1_t): 卷积核的形状
            stride (size_1_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_1_t | str): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_1_t): 卷积的输入步长
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            padding_mode (str): 边缘填充的方式
            affine (bool): 是否对标准化后的权重进行仿射操作
            eps (float): 方差偏移量
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


    def forward_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = _F.conv1d(o, self.weight_std, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Conv2d(Synapse, nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _Union[_size_2_t, str] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        二维卷积操作，输入一个大小为[B, C_{in}, H_{in}, W_{in}]的张量，输出一个大小为[B, C_{out}, H_{out}, W_{out}]的张量。
        Args:
            in_channels (int): 输入的频道数C_{in}
            out_channels (int): 输出的频道C_{out}
            kernel_size (size_2_t): 卷积核的形状
            stride (size_2_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_2_t | str): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_2_t): 卷积的输入步长
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            padding_mode (str): 边缘填充的方式
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(self)
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
        return nn.Conv2d.extra_repr(self) + ((", " + Synapse.extra_repr(self)) if len(Synapse.extra_repr(self)) else "")


    def forward_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = nn.Conv2d.forward(self, o)
        return x


class WSConv2d(_WeightStd, Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _Union[_size_2_t, str] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", affine: bool = True, eps: float = 1e-6, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        权重标准化的二维卷积操作。
        Args:
            in_channels (int): 输入的频道数C_{in}
            out_channels (int): 输出的频道C_{out}
            kernel_size (size_2_t): 卷积核的形状
            stride (size_2_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_2_t | str): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_2_t): 卷积的输入步长
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            padding_mode (str): 边缘填充的方式
            affine (bool): 是否对标准化后的权重进行仿射操作
            eps (float): 方差偏移量
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


    def forward_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = _F.conv2d(o, self.weight_std, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Conv3d(Synapse, nn.Conv3d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t = 1, padding: _Union[_size_3_t, str] = 0, dilation: _size_3_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        三维卷积操作，输入一个大小为[B, C_{in}, H_{in}, W_{in}, L_{in}]的张量，输出一个大小为[B, C_{out}, H_{out}, W_{out}, L_{out}]的张量。
        Args:
            in_channels (int): 输入的频道数C_{in}
            out_channels (int): 输出的频道C_{out}
            kernel_size (size_3_t): 卷积核的形状
            stride (size_3_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_3_t | str): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_3_t): 卷积的输入步长
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            padding_mode (str): 边缘填充的方式
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(self)
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
        return nn.Conv3d.extra_repr(self) + ((", " + Synapse.extra_repr(self)) if len(Synapse.extra_repr(self)) else "")


    def forward_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = nn.Conv3d.forward(self, o)
        return x


class WSConv3d(_WeightStd, Conv3d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t = 1, padding: _Union[_size_3_t, str] = 0, dilation: _size_3_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", affine: bool = True, eps: float = 1e-6, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        权重标准化的三维卷积操作。
        Args:
            in_channels (int): 输入的频道数C_{in}
            out_channels (int): 输出的频道C_{out}
            kernel_size (size_3_t): 卷积核的形状
            stride (size_3_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_3_t | str): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            dilation (size_3_t): 卷积的输入步长
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            padding_mode (str): 边缘填充的方式
            affine (bool): 是否对标准化后的权重进行仿射操作
            eps (float): 方差偏移量
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


    def forward_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = _F.conv3d(o, self.weight_std, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class ConvTranspose1d(Synapse, nn.ConvTranspose1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: _size_1_t = 0, output_padding: _size_1_t = 0, groups: int = 1, bias: bool = True, dilation: _size_1_t = 1, padding_mode: str = "zeros", device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        一维逆卷积操作，输入一个大小为[B, C_{in}, L_{in}]的张量，输出一个大小为[B, C_{out}, L_{out}]的张量。
        Args:
            in_channels (int): 输入的频道数C_{in}
            out_channels (int): 输出的频道C_{out}
            kernel_size (size_1_t): 卷积核的形状
            stride (size_1_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_1_t): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            output_padding (size_1_t): 在输出边缘填充的量
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            dilation (size_1_t): 卷积的输出步长
            padding_mode (str): 边缘填充的方式
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(self)
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
        return nn.ConvTranspose1d.extra_repr(self) + ((", " + Synapse.extra_repr(self)) if len(Synapse.extra_repr(self)) else "")


    def forward_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = nn.ConvTranspose1d.forward(self, o)
        return x


class ConvTranspose2d(Synapse, nn.ConvTranspose2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t = 0, output_padding: _size_2_t = 0, groups: int = 1, bias: bool = True, dilation: _size_2_t = 1, padding_mode: str = "zeros", device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        二维逆卷积操作，输入一个大小为[B, C_{in}, H_{in}, W_{in}]的张量，输出一个大小为[B, C_{out}, H_{out}, W_{out}]的张量。
        Args:
            in_channels (int): 输入的频道数C_{in}
            out_channels (int): 输出的频道C_{out}
            kernel_size (size_2_t): 卷积核的形状
            stride (size_2_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_2_t): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            output_padding (size_2_t): 在输出边缘填充的量
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            dilation (size_2_t): 卷积的输出步长
            padding_mode (str): 边缘填充的方式
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(self)
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
        return nn.ConvTranspose2d.extra_repr(self) + ((", " + Synapse.extra_repr(self)) if len(Synapse.extra_repr(self)) else "")


    def forward_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = nn.ConvTranspose2d.forward(self, o)
        return x


class ConvTranspose3d(Synapse, nn.ConvTranspose3d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t = 1, padding: _size_3_t = 0, output_padding: _size_3_t = 0, groups: int = 1, bias: bool = True, dilation: _size_3_t = 1, padding_mode: str = "zeros", device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        三维逆卷积操作，输入一个大小为[B, C_{in}, H_{in}, W_{in}, L_{in}]的张量，输出一个大小为[B, C_{out}, H_{out}, W_{out}, L_{out}]的张量。
        Args:
            in_channels (int): 输入的频道数C_{in}
            out_channels (int): 输出的频道C_{out}
            kernel_size (size_3_t): 卷积核的形状
            stride (size_3_t): 卷积的输出步长，决定卷积输出的形状
            padding (size_3_t): 在边缘填充的量（一般为卷积核大小的一半，向下取整）
            output_padding (size_3_t): 在输出边缘填充的量
            groups (int): 分组进行卷积操作的组数
            bias (bool): 是否要加入偏置
            dilation (size_3_t): 卷积的输出步长
            padding_mode (str): 边缘填充的方式
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(self)
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
        return nn.ConvTranspose3d.extra_repr(self) + ((", " + Synapse.extra_repr(self)) if len(Synapse.extra_repr(self)) else "")


    def forward_step(self, o: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            o (torch.Tensor): 来自上一层的输入脉冲$O_{j}^{l-1}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = nn.ConvTranspose3d.forward(self, o)
        return x


class _BatchNorm(Synapse, nn.modules.batchnorm._NormBase):
    def _init_params(self, training: bool, track_running_stats: bool, running_mean: torch.Tensor, running_var: torch.Tensor, momentum: float, num_batches_tracked: torch.Tensor) -> _Tuple[torch.Tensor, torch.Tensor, float, bool]:
        running_mean = running_mean if not training or track_running_stats else None
        running_var = running_var if not training or track_running_stats else None
        exponential_average_factor = 0.0 if momentum is None else momentum
        if training and track_running_stats:
            if num_batches_tracked is not None:
                num_batches_tracked.add_(1)
                exponential_average_factor = 1.0 / float(num_batches_tracked) if momentum is None else momentum
        bn_training = True if training else ((running_mean is None) and (running_var is None))
        return running_mean, running_var, exponential_average_factor, bn_training


    @staticmethod
    @torch.jit.script
    def _forward_steps(x: torch.Tensor, running_mean: _Optional[torch.Tensor], running_var: _Optional[torch.Tensor], weight: _Optional[torch.Tensor], bias: _Optional[torch.Tensor], bn_training: bool, momentum: float, eps: float) -> torch.Tensor:
        time_steps = x.shape[0]
        out = [_F.batch_norm(x[t], running_mean, running_var, weight, bias, bn_training, momentum, eps) for t in range(time_steps)]
        y = torch.stack(out)
        return y


    def forward_steps(self, x: torch.Tensor) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        self._check_input_dim(x)
        running_mean, running_var, exponential_average_factor, bn_training = self._init_params(self.training, self.track_running_stats, self.running_mean, self.running_var, self.momentum, self.num_batches_tracked)
        x = _BatchNorm._forward_steps(x, running_mean, running_var, self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
        return x


class BatchNorm1d(_BatchNorm, nn.BatchNorm1d):
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        一维批归一化。
        Args:
            num_features (int): 需要被归一化那一维度的长度
            eps (float): 参数epsilon
            momentum (float): 动量参数
            affine (bool): 是否启用参数gamma和beta，进行仿射变换
            track_running_stats (bool): 是否需要跟踪整个训练过程来进行批归一化的学习
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        _BatchNorm.__init__(self)
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


    def _check_input_dim(self, input: torch.Tensor):
        """
        Batchnorm中重要的部分，检查输入维度。
        Args:
            input (torch.Tensor): 输入张量
        """
        if self.multi_step_mode:
            supported_ndims = (3, 4)
        else:
            supported_ndims = (2, 3)
        if input.ndim not in supported_ndims:
            raise ValueError("expected %s input (got %dD input)" % (" or ".join(["%dD" % (d,) for d in supported_ndims]), input.ndim,))


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = nn.BatchNorm1d.forward(self, x)
        return x


class BatchNorm2d(_BatchNorm, nn.BatchNorm2d):
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        二维批归一化。
        Args:
            num_features (int): 需要被归一化那一维度的长度
            eps (float): 参数epsilon
            momentum (float): 动量参数
            affine (bool): 是否启用参数gamma和beta，进行仿射变换
            track_running_stats (bool): 是否需要跟踪整个训练过程来进行批归一化的学习
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        _BatchNorm.__init__(self)
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
        return nn.BatchNorm2d.extra_repr(self) + ((", " + Synapse.extra_repr(self)) if len(Synapse.extra_repr(self)) else "")


    def _check_input_dim(self, input: torch.Tensor):
        """
        Batchnorm中重要的部分，检查输入维度。
        Args:
            input (torch.Tensor): 输入张量
        """
        if self.multi_step_mode:
            supported_ndims = (5,)
        else:
            supported_ndims = (4,)
        if input.ndim not in supported_ndims:
            raise ValueError("expected %s input (got %dD input)" % (" or ".join(["%dD" % (d,) for d in supported_ndims]), input.ndim,))


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = nn.BatchNorm2d.forward(self, x)
        return x


class BatchNorm3d(_BatchNorm, nn.BatchNorm3d):
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        三维批归一化。
        Args:
            num_features (int): 需要被归一化那一维度的长度
            eps (float): 参数epsilon
            momentum (float): 动量参数
            affine (bool): 是否启用参数gamma和beta，进行仿射变换
            track_running_stats (bool): 是否需要跟踪整个训练过程来进行批归一化的学习
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        _BatchNorm.__init__(self)
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
        return nn.BatchNorm3d.extra_repr(self) + ((", " + Synapse.extra_repr(self)) if len(Synapse.extra_repr(self)) else "")


    def _check_input_dim(self, input: torch.Tensor):
        """
        Batchnorm中重要的部分，检查输入维度。
        Args:
            input (torch.Tensor): 输入张量
        """
        if self.multi_step_mode:
            supported_ndims = (6,)
        else:
            supported_ndims = (5,)
        if input.ndim not in supported_ndims:
            raise ValueError("expected %s input (got %dD input)" % (" or ".join(["%dD" % (d,) for d in supported_ndims]), input.ndim,))


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = nn.BatchNorm3d.forward(self, x)
        return x


class LayerNorm(Synapse, nn.LayerNorm):
    def __init__(self, normalized_shape: _shape_t, eps: float = 0.00001, elementwise_affine: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        数据归一化。
        Args:
            normalized_shape (shape_t): 在什么数据尺度上进行归一化
            eps (float): 参数epsilon
            elementwise_affine (bool): 是否启用参数gamma和beta，进行仿射变换
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        Synapse.__init__(self)
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
        return nn.LayerNorm.extra_repr(self) + ((", " + Synapse.extra_repr(self)) if len(Synapse.extra_repr(self)) else "")


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = nn.LayerNorm.forward(self, x)
        return x


class NormPlaceholder(Synapse):
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        归一化占位符，用于ANN转SNN的替换。
        Args:
            num_features (int): 需要被归一化的长度
            eps (float): 参数epsilon
            momentum (float): 动量参数
            affine (bool): 是否启用参数gamma和beta，进行仿射变换
            track_running_stats (bool): 是否需要跟踪整个训练过程来进行批归一化的学习
            device (torch.device): 所计算的设备
            dtype (torch.dtype): 所计算的数据类型
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_features, device = device, dtype = dtype))
            self.bias = nn.Parameter(torch.empty(num_features, device = device, dtype = dtype))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, device = device, dtype = dtype))
            self.register_buffer('running_var', torch.ones(num_features, device = device, dtype = dtype))
            self.register_buffer('num_batches_tracked', torch.tensor(0, device = device, dtype = torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)


    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        return x.clone()


class Identity(Synapse, nn.Identity):
    def __init__(self) -> None:
        """
        同一层，输出为输入。
        """
        Synapse.__init__(self)
        nn.Identity.__init__(self)


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return nn.Identity.extra_repr(self) + ((", " + Synapse.extra_repr(self)) if len(Synapse.extra_repr(self)) else "")
    

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        Returns:
            x (torch.Tensor): 突触的突触后电位$X_{i}^{l}(t)$
        """
        x = nn.Identity.forward(self, x)
        return x


class MultiheadAttention(Synapse, nn.MultiheadAttention):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True, add_bias_kv: bool = False, add_zero_attn: bool = False, kdim: _Optional[int] = None, vdim: _Optional[int] = None, batch_first: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        Synapse.__init__(self)
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


    def forward_step(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: _Optional[torch.Tensor] = None, need_weights: bool = True, attn_mask: _Optional[torch.Tensor] = None, average_attn_weights: bool = True) -> _Tuple[torch.Tensor, _Optional[torch.Tensor]]:
        return nn.MultiheadAttention.forward(self, query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights)


    def forward_steps(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: _Optional[torch.Tensor] = None, need_weights: bool = True, attn_mask: _Optional[torch.Tensor] = None, average_attn_weights: bool = True) -> _Tuple[torch.Tensor, _Optional[torch.Tensor]]:
        is_tensor = lambda x: x is not None and isinstance(x, torch.Tensor)
        T = value.shape[0]
        reshape = lambda x, dim: torch.stack([x] * T) if x.ndim < dim else x
        query = reshape(query, value.ndim)
        key = reshape(key, value.ndim)
        key_padding_mask = reshape(key_padding_mask, value.ndim - 1) if is_tensor(key_padding_mask) else attn_mask
        attn_mask = reshape(attn_mask, value.ndim) if is_tensor(attn_mask) else attn_mask
        B = 1
        if value.ndim > 3:
            if self.batch_first:
                B = value.shape[-3]
                merge_tb = lambda x: x.flatten(0, 1)
                split_tb = lambda x: x.reshape([T, B] + list(x.shape[1:]))
            else:
                B = value.shape[-2]
                merge_tb = lambda x: x.swapaxes(1, 2).flatten(0, 1).swapaxes(0, 1)
                split_tb = lambda x: x.reshape([x.shape[0], T, B] + list(x.shape[2:])).swapaxes(0, 1)
            merge_mask = lambda x: x.flatten(0, 1)
            split_wt = lambda x: x.reshape([T, B] + list(x.shape[1:]))
        else:
            if self.batch_first:
                merge_tb = lambda x: x
                split_tb = lambda x: x
            else:
                merge_tb = lambda x: x.swapaxes(0, 1)
                split_tb = lambda x: x.swapaxes(0, 1)
            merge_mask = lambda x: x
            split_wt = lambda x: x
        query = merge_tb(query)
        key = merge_tb(key)
        value = merge_tb(value)
        key_padding_mask = merge_mask(key_padding_mask) if is_tensor(key_padding_mask) else None
        attn_mask = merge_mask(attn_mask) if is_tensor(attn_mask) else None
        attn_output, attn_output_weights = nn.MultiheadAttention.forward(self, query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights)
        attn_output = split_tb(attn_output)
        attn_output_weights = split_wt(attn_output_weights) if is_tensor(attn_output_weights) else attn_output_weights
        return attn_output, attn_output_weights