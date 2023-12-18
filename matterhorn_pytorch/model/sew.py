# -*- coding: UTF-8 -*-
"""
Spiking Element-Wise (SEW) ResNet。
Reference:
[Fang W, Yu Z, Chen Y, et al. Deep residual learning in spiking neural networks\[J\]. Advances in Neural Information Processing Systems, 2021, 34: 21056-21069.]
(https://proceedings.neurips.cc/paper/2021/hash/afe434653a898da20044041262b3ac74-Abstract.html)
"""


import torch
import torch.nn as nn
import matterhorn_pytorch.snn as snn
from typing import Tuple, Callable
try:
    from rich import print
except:
    pass


class ResADD(snn.Module):
    def __init__(self) -> None:
        super().__init__(
            multi_time_step = True
        )


    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，ADD连接函数。
        Args:
            a (torch.Tensor): 经过两层卷积处理后的脉冲张量
            s (torch.Tensor): 原始脉冲张量
        Returns:
            o (torch.Tensor): 输出脉冲张量
        """
        return a + s


class ResAND(snn.Module):
    def __init__(self) -> None:
        super().__init__(
            multi_time_step = True
        )


    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，AND连接函数。
        Args:
            a (torch.Tensor): 经过两层卷积处理后的脉冲张量
            s (torch.Tensor): 原始脉冲张量
        Returns:
            o (torch.Tensor): 输出脉冲张量
        """
        return a * s


class ResIAND(snn.Module):
    def __init__(self) -> None:
        super().__init__(
            multi_time_step = True
        )


    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，IAND连接函数。
        Args:
            a (torch.Tensor): 经过两层卷积处理后的脉冲张量
            s (torch.Tensor): 原始脉冲张量
        Returns:
            o (torch.Tensor): 输出脉冲张量
        """
        return (1.0 - a) * s


def ConvLIF(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, tau_m: float = 2.0, spiking_function: snn.Module = snn.Rectangular(), trainable: bool = False):
    """
    绑定卷积神经元（突触）和LIF神经元（胞体）
    Args:
        in_channels (int): 输入脉冲通道数
        out_channels (int): 输出脉冲通道数
        kernel_size (int): 卷积核的大小
        stride (int): 卷积步长
        tau_m (float): 参数τ_{m}，神经元时间常数
        spiking_function (snn.Module): 脉冲函数
        trainable (bool): 参数是否可以训练
    """
    return snn.Sequential(
        snn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = kernel_size // 2
        ),
        snn.BatchNorm2d(
            num_features = out_channels
        ),
        snn.LIF(
            tau_m = tau_m,
            spiking_function = spiking_function,
            trainable = trainable
        )
    )


class SEWBlock(snn.Module):
    def __init__(self, in_channels: int, out_channels: int, tau_m: float = 2.0, spiking_function: snn.Module = snn.Rectangular(), residual_connection: snn.Module = ResADD(), down_sampling: bool = False, trainable: bool = False) -> None:
        """
        Spiking Element-Wise Block， SEW ResNet的单元。
        Args:
            in_channels (int): 输入脉冲通道数
            out_channels (int): 输出脉冲通道数
            tau_m (float): 参数τ_{m}，神经元时间常数
            spiking_function (snn.Module): 脉冲函数
            residual_connection (snn.Module): 脉冲连接方式
            down_sampling (bool): 是否进行下采样（出来的图像大小是原大小的一半）
            trainable (bool): 参数是否可以训练
        """
        super().__init__(
            multi_time_step = True
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvLIF(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 2 if down_sampling else 1,
            tau_m = tau_m,
            spiking_function = spiking_function,
            trainable = trainable
        )
        self.conv2 = ConvLIF(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            tau_m = tau_m,
            spiking_function = spiking_function,
            trainable = trainable
        )
        self.down_sampling = down_sampling
        if self.down_sampling:
            self.down_sampling_block = ConvLIF(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2,
                tau_m = tau_m,
                spiking_function = spiking_function,
                trainable = trainable
            )
        elif in_channels != out_channels:
            self.down_sampling_block = ConvLIF(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                tau_m = tau_m,
                spiking_function = spiking_function,
                trainable = trainable
            )
        else:
            self.down_sampling_block = None
        self.residual_connection = residual_connection
    

    def reset(self) -> None:
        """
        重置模型。
        """
        self.conv1.reset()
        self.conv2.reset()
        if self.down_sampling_block is not None:
            self.down_sampling_block.reset()
        self.residual_connection.reset()
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入脉冲张量
        Returns:
            o (torch.Tensor): 输出脉冲张量
        """
        a = self.conv2(self.conv1(x))
        if self.down_sampling_block is not None:
            s = self.down_sampling_block(x)
        else:
            s = x
        o = self.residual_connection(a, s)
        return o


class SEWRes18(snn.Module):
    def __init__(self, input_h_w: Tuple[int] = (128, 128), num_classes: int = 10, tau_m: float = 2.0, spiking_function: snn.Module = snn.Rectangular(), residual_connection: snn.Module = ResADD(), trainable: bool = False) -> None:
        """
        Spiking Element-Wise Block， SEW ResNet的单元。
        Args:
            input_h_w: Tuple[int] 输出脉冲通道数
            num_classes (int): 输出类别数
            tau_m (float): 参数τ_{m}，神经元时间常数
            spiking_function (snn.Module): 脉冲函数
            residual_connection (snn.Module): 脉冲连接方式
            trainable (bool): 参数是否可以训练
        """
        super().__init__(
            multi_time_step = True
        )
        self.model = snn.Sequential(
            snn.DirectEncoder(),
            ConvLIF(
                in_channels = 2,
                out_channels = 64,
                kernel_size = 3,
                stride = 2,
                tau_m = tau_m,
                spiking_function = spiking_function,
                trainable = trainable
            ), # 1, [T, 64, 64, 64]
            snn.MaxPool2d(
                kernel_size = 2
            ), # 2, [T, 64, 32, 32]
            SEWBlock(
                in_channels = 64,
                out_channels = 64,
                tau_m = tau_m,
                spiking_function = spiking_function,
                residual_connection = residual_connection,
                trainable = trainable
            ), # 3, 4, [T, 64, 32, 32]
            SEWBlock(
                in_channels = 64,
                out_channels = 64,
                tau_m = tau_m,
                spiking_function = spiking_function,
                residual_connection = residual_connection,
                trainable = trainable
            ), # 5, 6, [T, 64, 32, 32]
            SEWBlock(
                in_channels = 64,
                out_channels = 128,
                tau_m = tau_m,
                spiking_function = spiking_function,
                residual_connection = residual_connection,
                down_sampling = True,
                trainable = trainable
            ), # 7, 8, [T, 128, 16, 16]
            SEWBlock(
                in_channels = 128,
                out_channels = 128,
                tau_m = tau_m,
                spiking_function = spiking_function,
                residual_connection = residual_connection,
                trainable = trainable
            ), # 9, 10, [T, 128, 16, 16]
            SEWBlock(
                in_channels = 128,
                out_channels = 256,
                tau_m = tau_m,
                spiking_function = spiking_function,
                residual_connection = residual_connection,
                down_sampling = True,
                trainable = trainable
            ), # 11, 12, [T, 256, 8, 8]
            SEWBlock(
                in_channels = 256,
                out_channels = 256,
                tau_m = tau_m,
                spiking_function = spiking_function,
                residual_connection = residual_connection,
                trainable = trainable
            ), # 13, 14, [T, 256, 8, 8]
            SEWBlock(
                in_channels = 256,
                out_channels = 512,
                tau_m = tau_m,
                spiking_function = spiking_function,
                residual_connection = residual_connection,
                down_sampling = True,
                trainable = trainable
            ), # 15, 16, [T, 512, 4, 4]
            SEWBlock(
                in_channels = 512,
                out_channels = 512,
                tau_m = tau_m,
                spiking_function = spiking_function,
                residual_connection = residual_connection,
                trainable = trainable
            ), # 17, 18, [T, 512, 4, 4]
            snn.AvgSpikeDecoder(), # [512, 4, 4]
            nn.AvgPool2d(
                kernel_size = (input_h_w[0] >> 5, input_h_w[1] >> 5)
            ), # [512, 1, 1]
            nn.Flatten(), # [512]
            nn.Linear(
                in_features = 512,
                out_features = num_classes
            ), # [10]
        )
    

    def reset(self) -> None:
        """
        重置模型。
        """
        self.model.reset()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入脉冲张量
        Returns:
            x (torch.Tensor): 输出张量
        """
        x = self.model(x)
        return x