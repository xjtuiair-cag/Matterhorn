import torch
import torch.nn as nn
from matterhorn.snn import surrogate


"""
脉冲神经网络的整个神经元层，输入为脉冲，输出为脉冲。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
"""


class val_to_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.ge(0.5).to(x)
    

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class SRM0(nn.Module):
    def __init__(self, tau_m: float = 2.0, tau_s: float = 8.0, u_threshold: float = 1.0) -> None:
        super().__init__()
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.u_threshold = u_threshold


class MaxPool1d(nn.MaxPool1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return val_to_spike.apply(y)


class MaxPool2d(nn.MaxPool2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return val_to_spike.apply(y)


class MaxPool3d(nn.MaxPool3d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return val_to_spike.apply(y)


class AvgPool1d(nn.AvgPool1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return val_to_spike.apply(y)


class AvgPool2d(nn.AvgPool2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return val_to_spike.apply(y)


class AvgPool3d(nn.AvgPool3d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return val_to_spike.apply(y)


class Flatten(nn.Flatten):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return val_to_spike.apply(y)


class Unflatten(nn.Unflatten):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return val_to_spike.apply(y)