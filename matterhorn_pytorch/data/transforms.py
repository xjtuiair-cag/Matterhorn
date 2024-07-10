# -*- coding: UTF-8 -*-
"""
数据变换的操作（类）。
"""


import torch
import torch.nn as nn
from torchvision.transforms import Compose
import matterhorn_pytorch.data.functional as DF
from typing import Iterable


class Clip(nn.Module):
    def __init__(self, clip_range: slice) -> None:
        """
        裁剪事件序列。
        """
        super().__init__()
        self.clip_start = clip_range.start
        self.clip_end = clip_range.stop


    def extra_repr(self) -> str:
        return ", ".join(["start=%d" % self.clip_start, "end=%d" % self.clip_end])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[(x[:, 0] > self.clip_start) & (x[:, 0] < self.clip_end)]


class Sampling(nn.Module):
    def __init__(self, span: slice) -> None:
        """
        对事件序列进行采样。
        """
        super().__init__()
        self.sampling_span = span


    def extra_repr(self) -> str:
        return ", ".join(["span=%d" % self.sampling_span])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[::self.sampling_span]