# -*- coding: UTF-8 -*-
"""
液体状态机的输入和输出模块。
"""


import torch
import torch.nn as nn
import matterhorn_pytorch.snn as snn
from matterhorn_pytorch.lsm.functional import init_adjacent_direct
from typing import Iterable
try:
    from rich import print
except:
    pass


class Cast(snn.Module):
    def __init__(self, in_features: int, out_features: int, in_indices: Iterable[int], out_indices: Iterable[int], multi_time_step: bool = True) -> None:
        """
        LSM的输入/输出形状转化，将数据投射至正确的神经元输入上。
        Args:
            in_features (int): 输入长度
            out_features (int): 输出长度
            in_indices (int*): 输入索引
            out_indices (int*): 输出索引
            multi_time_step (bool): 是否调整为多个时间步模式
        """
        super().__init__(
            multi_time_step = multi_time_step
        )
        self.in_features = in_features
        self.out_features = out_features
        self.in_indices = list(in_indices)
        self.out_indices = list(out_indices)
        feature_num = max(in_features, out_features)
        adjacent = init_adjacent_direct(feature_num, in_indices, out_indices)
        adjacent = adjacent[:in_features,:out_features]
        self.adjacent = nn.Parameter(adjacent, requires_grad = False)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 当前输入
        Returns:
            y (torch.Tensor): 当前输出
        """
        return nn.functional.linear(x, self.adjacent.T, None)