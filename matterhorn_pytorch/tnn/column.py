# -*- coding: UTF-8 -*-
"""
TNN的柱，包含兴奋柱、抑制柱与认知柱等。可以看作是组成TNN的层的基本结构。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.snn import Module as _Module


class Lateral(_Module):
    def __init__(self, tl: int, kl: int) -> None:
        """
        TNN的侧抑制柱
        """
        super().__init__()
        self.tl = tl
        self.kl = kl
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass