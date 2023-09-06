# -*- coding: UTF-8 -*-
"""
Spiking Element-Wise (SEW) ResNet。
Reference:
[Fang W, Yu Z, Chen Y, et al. Deep residual learning in spiking neural networks[J]. Advances in Neural Information Processing Systems, 2021, 34: 21056-21069.]
(https://proceedings.neurips.cc/paper/2021/hash/afe434653a898da20044041262b3ac74-Abstract.html)
"""


import torch
import torch.nn as nn
import matterhorn.snn as snn
torch.autograd.set_detect_anomaly(True)
try:
    from rich import print
except:
    pass


class ResADD(snn.Module):
    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return a + s


class ResAND(snn.Module):
    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return a * s


class ResIAND(snn.Module):
    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return (1.0 - a) * s


def ConvLIF(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, tau_m: float = 2.0, spiking_function: snn.Module = snn.Rectangular()):
    return snn.SpatialContainer(
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
            spiking_function = spiking_function
        )
    )


class SEWBlock(snn.Module):
    def __init__(self, in_channels: int, out_channels: int, tau_m: float = 2.0, spiking_function: snn.Module = snn.Rectangular(), residual_connection: snn.Module = ResADD(), down_sampling: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvLIF(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 2 if down_sampling else 1,
            tau_m = tau_m,
            spiking_function = spiking_function
        )
        self.conv2 = ConvLIF(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            tau_m = tau_m,
            spiking_function = spiking_function
        )
        self.residual_connection = residual_connection
        self.down_sampling = down_sampling
        if down_sampling:
            self.down_sampling_block = ConvLIF(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2,
                tau_m = tau_m
            )
        else:
            self.down_sampling_block = None
    

    def reset(self) -> None:
        self.conv1.reset()
        self.conv2.reset()
        self.residual_connection.reset()
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv2(self.conv1(x))
        if self.down_sampling:
            s = self.down_sampling_block(x)
        else:
            s = x
        s = self.residual_connection(a, s)
        return s


class SEWRes18(snn.Module):
    def __init__(self, input_h_w = (128, 128), tau_m = 2.0, spiking_function: snn.Module = snn.Rectangular(), residual_connection: snn.Module = ResADD()) -> None:
        super().__init__()
        self.snn_model = snn.SNNContainer(
            encoder = snn.DirectEncoder(),
            snn_model = snn.TemporalContainer(
                snn.SpatialContainer(
                    ConvLIF(
                        in_channels = 2,
                        out_channels = 64,
                        kernel_size = 3,
                        stride = 2,
                        tau_m = tau_m,
                        spiking_function = spiking_function
                    ), # 1, [T, 64, 64, 64]
                    snn.MaxPool2d(
                        kernel_size = 2
                    ), # 2, [T, 64, 32, 32]
                    SEWBlock(
                        in_channels = 64,
                        out_channels = 64,
                        tau_m = tau_m,
                        spiking_function = spiking_function,
                        residual_connection = residual_connection
                    ), # 3, 4, [T, 64, 32, 32]
                    SEWBlock(
                        in_channels = 64,
                        out_channels = 64,
                        tau_m = tau_m,
                        spiking_function = spiking_function,
                        residual_connection = residual_connection
                    ), # 5, 6, [T, 64, 32, 32]
                    SEWBlock(
                        in_channels = 64,
                        out_channels = 128,
                        tau_m = tau_m,
                        spiking_function = spiking_function,
                        residual_connection = residual_connection,
                        down_sampling = True
                    ), # 7, 8, [T, 128, 16, 16]
                    SEWBlock(
                        in_channels = 128,
                        out_channels = 128,
                        tau_m = tau_m,
                        spiking_function = spiking_function,
                        residual_connection = residual_connection
                    ), # 9, 10, [T, 128, 16, 16]
                    SEWBlock(
                        in_channels = 128,
                        out_channels = 256,
                        tau_m = tau_m,
                        spiking_function = spiking_function,
                        residual_connection = residual_connection,
                        down_sampling = True
                    ), # 11, 12, [T, 256, 8, 8]
                    SEWBlock(
                        in_channels = 256,
                        out_channels = 256,
                        tau_m = tau_m,
                        spiking_function = spiking_function,
                        residual_connection = residual_connection
                    ), # 13, 14, [T, 256, 8, 8]
                    SEWBlock(
                        in_channels = 256,
                        out_channels = 512,
                        tau_m = tau_m,
                        spiking_function = spiking_function,
                        residual_connection = residual_connection,
                        down_sampling = True
                    ), # 15, 16, [T, 512, 4, 4]
                    SEWBlock(
                        in_channels = 512,
                        out_channels = 512,
                        tau_m = tau_m,
                        spiking_function = spiking_function,
                        residual_connection = residual_connection
                    ) # 17, 18, [T, 512, 4, 4]
                ),
                reset_after_process = False
            ),
            decoder = snn.AvgSpikeDecoder()
        ) # [512, 4, 4]
        self.ann_model = nn.Sequential(
            nn.AvgPool2d(
                kernel_size = (input_h_w[0] >> 5, input_h_w[1] >> 5)
            ), # [512, 1, 1]
            nn.Flatten(), # [512]
            nn.Linear(
                in_features = 512,
                out_features = 10
            ),
            nn.ReLU() # [10]
        )
    

    def reset(self) -> None:
        self.snn_model.reset()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.snn_model(x)
        print("1", x.shape)
        x = self.ann_model(x)
        print("2", x.shape)
        return x