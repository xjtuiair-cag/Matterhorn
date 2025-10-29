# -*- coding: U_TF-8 -*-
"""
TNN的组件，构建TNN兴奋柱和抑制柱的基本组件。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.snn import Module as _Module
import matterhorn_pytorch.tnn.functional as _TF


class Bitonic(_Module):
    def __init__(self, level: int, asc: bool = True) -> None:
        """
        双调排序网络。通过时空代数的min和max算子对时空信号进行排序操作。
        Args:
            level (int): 排序的级别l，输入为形状为[T, B, C, 2^l]的脉冲序列
            asc (bool): 是否为升序排序，为True则为升序，否则为降序
        """
        super().__init__()
        self.level = level
        self.asc = asc
        self.length = 2 ** self.level
        self.half_length = 2 ** (self.level - 1)
        if self.level > 1:
            self.asc_unit = Bitonic(self.level - 1, asc = True)
            self.desc_unit = Bitonic(self.level - 1, asc = False)
        else:
            self.asc_unit = None
            self.desc_unit = None
    

    def forward(self, x: torch.Tensor, post: bool = False) -> torch.Tensor:
        """
        双调网络的前向传播函数。
        Args:
            x (torch.Tensor): 输入信号x，形状为[T, B, C, 2^l]
            post (bool): 是否在大排序之后调用，设为False即可
        Returns:
            y (torch.Tensor): 输出信号y，形状为[T, B, C, 2^l]
        """
        if self.level <= 0 or x.shape[-1] <= 1:
            y = x
        elif self.level == 1:
            y0: torch.Tensor
            y1: torch.Tensor
            y0 = _TF.s_min(x[..., 0], x[..., 1])
            y1 = _TF.s_max(x[..., 0], x[..., 1])
            if self.asc:
                y = torch.stack([y0, y1])
            else:
                y = torch.stack([y1, y0])
            y_dims = list(range(y.ndim))
            y_dims = y_dims[1:] + [0]
            y = y.permute(*y_dims)
        else:
            if x.shape[-1] <= self.half_length:
                if self.asc:
                    y = self.asc_unit(x, post = True)
                else:
                    y = self.desc_unit(x, post = True)
            else:
                y0: torch.Tensor
                y1: torch.Tensor
                y2: torch.Tensor
                y3: torch.Tensor
                y0 = x[..., :self.half_length]
                y1 = x[..., self.half_length:]
                if not post:
                    y0 = self.asc_unit(y0)
                    y1 = self.desc_unit(y1)
                if self.asc:
                    y2 = _TF.s_min(y0, y1)
                    y3 = _TF.s_max(y0, y1)
                    y0 = self.asc_unit(y2, post = True)
                    y1 = self.asc_unit(y3, post = True)
                else:
                    y2 = _TF.s_max(y0, y1)
                    y3 = _TF.s_min(y0, y1)
                    y0 = self.desc_unit(y2, post = True)
                    y1 = self.desc_unit(y3, post = True)
                y = torch.cat([y0, y1], dim = y0.ndim - 1)
        return y


class Firing(_Module):
    def __init__(self, u_threshold: int) -> None:
        """
        统计上升/下降的时间，并发射脉冲。
        """
        super().__init__()
        self.u_threshold = nn.Parameter(torch.tensor(u_threshold), requires_grad = False)
    

    def forward(self, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        统计单元的前向传播函数。
        Args:
            u (torch.Tensor): 上升脉冲序列（按时间由小到大排序），形状为[T, B, C, Q]
            d (torch.Tensor): 下降脉冲序列（按时间由小到大排序），形状为[T, B, C, Q]
        Returns:
            out (torch.Tensor): 结果（是否达到阈值并发射脉冲），形状为[T, B, C, 1]
        """
        q = u.shape[-1]
        upper_threshold = q - self.u_threshold + 1
        u = u[..., -upper_threshold:]
        d = d[..., :upper_threshold]
        res = _TF.s_lt(u, d)
        out = res[..., 0:1]
        for i in range(1, res.shape[-1]):
            out = _TF.s_min(out, res[..., i:i + 1])
        return out