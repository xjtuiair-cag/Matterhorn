# -*- coding: UTF-8 -*-
"""
脉冲簇（Volley）编解码与时空变换工具。

参考文献：Smith J E. Space-Time Computing with Temporal Neural Networks[M].
Morgan & Claypool, 2017. 第6.2节（脉冲编码）与第7.3节（OnOff编码）。
"""


import torch
import torch.nn as nn
from typing import Tuple as _Tuple, Optional as _Optional


# 无脉冲线的哨兵值（书中记为 "-" 或 ∞）
EMPTY_SPIKE: float = float("inf")


def is_spike_present(t: torch.Tensor) -> torch.Tensor:
    """
    判断时间戳张量中哪些位置存在有效脉冲。
    Args:
        t (torch.Tensor): 脉冲首次发放时间戳
    Returns:
        mask (torch.Tensor): 布尔掩码，True 表示该线有脉冲
    """
    return torch.isfinite(t)


def normalize_time_volley(t: torch.Tensor) -> _Tuple[torch.Tensor, torch.Tensor]:
    """
    将时间簇归一化到局部时间参考系（最早脉冲时刻为 0）。
    对应 Smith 书第7.4.1节评估模式中的 normalize 步骤。
    Args:
        t (torch.Tensor): 时间簇，形状 [..., M]，无脉冲处为 EMPTY_SPIKE
    Returns:
        t_norm (torch.Tensor): 归一化后的时间簇
        t_min (torch.Tensor): 各簇最早脉冲的全局时间，形状为 t[..., :1] 广播前的缩减维
    """
    present = is_spike_present(t)
    safe = t.masked_fill(~present, EMPTY_SPIKE)
    t_min = safe.min(dim = -1, keepdim = True).values
    t_norm = t - t_min
    t_norm = t_norm.masked_fill(~present, EMPTY_SPIKE)
    return t_norm, t_min.squeeze(-1)


def denormalize_time_volley(t_norm: torch.Tensor, t_min: torch.Tensor) -> torch.Tensor:
    """
    将局部时间簇还原为全局时间参考系。
    Args:
        t_norm (torch.Tensor): 归一化时间簇，形状 [..., M]
        t_min (torch.Tensor): 最早脉冲全局时间，形状 [...] 或 [..., 1]
    Returns:
        t (torch.Tensor): 全局时间簇
    """
    if t_min.ndim < t_norm.ndim:
        t_min = t_min.unsqueeze(-1)
    present = is_spike_present(t_norm)
    t = t_norm + t_min
    return t.masked_fill(~present, EMPTY_SPIKE)


def value_to_time_volley(v: torch.Tensor, t_max: float, v_max: float) -> torch.Tensor:
    """
    值簇转时间簇：vi = α(Tmax - ti)，即 ti = Tmax - vi/α，α = Vmax/Tmax。
    Args:
        v (torch.Tensor): 值簇，形状 [..., M]，无值处为 EMPTY_SPIKE
        t_max (float): 最大时间窗 Tmax
        v_max (float): 最大值 Vmax
    Returns:
        t (torch.Tensor): 时间簇
    """
    present = is_spike_present(v)
    alpha = v_max / t_max
    t = t_max - v / alpha
    return t.masked_fill(~present, EMPTY_SPIKE)


def time_to_value_volley(t: torch.Tensor, t_max: float, v_max: float) -> torch.Tensor:
    """
    时间簇转值簇：vi = α(Tmax - ti)。
    Args:
        t (torch.Tensor): 时间簇
        t_max (float): Tmax
        v_max (float): Vmax
    Returns:
        v (torch.Tensor): 值簇
    """
    present = is_spike_present(t)
    alpha = v_max / t_max
    v = alpha * (t_max - t)
    return v.masked_fill(~present, EMPTY_SPIKE)


def merge_time_volleys(*volleys: torch.Tensor) -> torch.Tensor:
    """
    沿最后一维拼接多个时间簇（对应 CC 输入 bundle 合并）。
    Args:
        *volleys (torch.Tensor): 若干时间簇，batch 维需一致
    Returns:
        merged (torch.Tensor): 合并后的时间簇
    """
    return torch.cat(volleys, dim = -1)


def onoff_encode(values: torch.Tensor, v_max: float) -> torch.Tensor:
    """
    OnOff 编码：将原始特征向量与其负片拼接，消除簇间偏序关系。
    对应 Smith 书第7.3节：xm+k = Imax - xk。
    Args:
        values (torch.Tensor): 原始值向量，形状 [..., M]，取值 [0, v_max]
        v_max (float): 最大值 Imax/Vmax
    Returns:
        encoded (torch.Tensor): OnOff 编码后的向量，形状 [..., 2M]
    """
    neg = v_max - values
    return torch.cat([values, neg], dim = -1)


class OnOffEncoder(nn.Module):
    """
    OnOff 编码模块：将模拟值向量编码为可转为时间簇的 OnOff 值向量。
    """

    def __init__(self, v_max: float = 1.0) -> None:
        """
        Args:
            v_max (float): 特征最大值 Vmax/Imax
        """
        super().__init__()
        self.v_max = v_max


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征，形状 [B, M]
        Returns:
            y (torch.Tensor): OnOff 编码特征，形状 [B, 2M]
        """
        return onoff_encode(x, self.v_max)
