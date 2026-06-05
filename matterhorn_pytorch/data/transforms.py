# -*- coding: UTF-8 -*-
"""
数据变换的操作（类）。
"""


import torch
import torch.nn as nn
import torchvision.transforms as _transforms
import matterhorn_pytorch.snn.functional as _SF
import matterhorn_pytorch.data.functional as _DF


class Clip(nn.Module):
    def __init__(self, clip_range: slice) -> None:
        """
        裁剪事件序列。
        """
        super().__init__()
        self.clip_start = clip_range.start
        self.clip_end = clip_range.stop


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["start=%d" % self.clip_start, "end=%d" % self.clip_end])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        按时间范围裁剪事件序列。
        Args:
            x (torch.Tensor): 事件序列，第0列为时间
        Returns:
            y (torch.Tensor): 裁剪后的事件序列
        """
        return x[(x[:, 0] > self.clip_start) & (x[:, 0] < self.clip_end)]


class Sampling(nn.Module):
    def __init__(self, span: slice) -> None:
        """
        对事件序列进行采样。
        """
        super().__init__()
        self.sampling_span = span


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return ", ".join(["span=%d" % self.sampling_span])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对事件序列按固定步长采样。
        Args:
            x (torch.Tensor): 事件序列
        Returns:
            y (torch.Tensor): 采样后的事件序列
        """
        return x[::self.sampling_span]