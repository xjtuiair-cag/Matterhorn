# -*- coding: UTF-8 -*-
"""
数据变换的操作（类）。
"""


import torch
import torch.nn as nn
import torchvision.transforms as _tf
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


class TemporalCompose(_tf.Compose):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x, tb = _SF.merge_time_steps_batch_size(x)
        for t in self.transforms:
            x = t(x)
        x = _SF.split_time_steps_batch_size(x, tb)
        return x