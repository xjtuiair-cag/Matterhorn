# -*- coding: UTF-8 -*-
"""
抑制柱（Inhibitory Column）模型。

实现 Smith 书第7.5节的前馈抑制（FFI）与侧向抑制（LI / tk-WTA）。
抑制柱在更高抽象层操作，不建模单个抑制神经元，而是对脉冲簇进行过滤。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.snn.skeleton import Module as _Module
from matterhorn_pytorch.tnn import volley as _TV
from typing import Tuple as _Tuple, Optional as _Optional


def feedforward_inhibit(
    t_volley: torch.Tensor,
    t_f: float,
    k_f: int
) -> torch.Tensor:
    """
    前馈抑制 FFI（Smith 书第7.5.3节）。
    若在首个脉冲后 t_f 时间窗内超过 k_f 个脉冲，则整簇清零（nullify）。
    Args:
        t_volley (torch.Tensor): 归一化时间簇 [..., M]
        t_f (float): 时间窗参数 tF
        k_f (int): 脉冲数阈值 kF
    Returns:
        out (torch.Tensor): 抑制后的时间簇
    """
    present = _TV.is_spike_present(t_volley)
    in_window = present & (t_volley <= t_f)
    count = in_window.sum(dim = -1, keepdim = True)
    keep = count <= k_f
    out = t_volley.clone()
    out = out.masked_fill(~keep.expand_as(out), _TV.EMPTY_SPIKE)
    out = out.masked_fill(~present, _TV.EMPTY_SPIKE)
    return out


def lateral_inhibit(
    t_volley: torch.Tensor,
    t_l: float,
    k_l: int
) -> torch.Tensor:
    """
    侧向抑制 LI / tk-WTA（Smith 书第7.5.2节）。
    保留时间 ≤ t_l 的最早 k_l 个脉冲，其余置空。
    Args:
        t_volley (torch.Tensor): 归一化时间簇 [..., M]
        t_l (float): 时间窗 tL
        k_l (int): 保留脉冲数 kL
    Returns:
        out (torch.Tensor): 抑制后的时间簇
    """
    shape = t_volley.shape
    flat = t_volley.reshape(-1, shape[-1])
    batch = flat.shape[0]
    out = torch.full_like(flat, _TV.EMPTY_SPIKE)

    for b in range(batch):
        row = flat[b]
        present = _TV.is_spike_present(row)
        if not present.any():
            continue
        valid_t = row[present]
        valid_idx = present.nonzero(as_tuple = False).squeeze(-1)
        in_window = valid_t <= t_l
        if not in_window.any():
            continue
        candidates = valid_t[in_window]
        cand_idx = valid_idx[in_window]
        order = candidates.argsort()
        keep_n = min(k_l, order.numel())
        selected = cand_idx[order[:keep_n]]
        out[b, selected] = row[selected]

    return out.reshape(shape)


class FeedforwardInhibition(_Module):
    """
    前馈抑制柱（FFI）。
    """

    def __init__(self, t_f: float = 1.3, k_f: int = 10) -> None:
        """
        Args:
            t_f (float): 时间窗 tF（毫秒或抽象时间步）
            k_f (int): 脉冲数阈值 kF
        """
        super().__init__()
        self.t_f = t_f
        self.k_f = k_f


    def extra_repr(self) -> str:
        return "t_f=%g, k_f=%d" % (self.t_f, self.k_f)


    def forward(self, t_volley: torch.Tensor, normalized: bool = True) -> torch.Tensor:
        """
        Args:
            t_volley (torch.Tensor): 时间簇 [B, M]
            normalized (bool): 是否为归一化时间簇（FFI 在归一化参考系下工作）
        Returns:
            out (torch.Tensor): 抑制后的时间簇
        """
        t_in = t_volley
        t_min = None
        if not normalized:
            t_in, t_min = _TV.normalize_time_volley(t_volley)
        out = feedforward_inhibit(t_in, self.t_f, self.k_f)
        if not normalized and t_min is not None:
            present = _TV.is_spike_present(out)
            out = out + t_min.unsqueeze(-1)
            out = out.masked_fill(~present, _TV.EMPTY_SPIKE)
        return out


class LateralInhibition(_Module):
    """
    侧向抑制柱（LI）。
    """

    def __init__(self, t_l: float = 5.0, k_l: int = 3) -> None:
        """
        Args:
            t_l (float): 时间窗 tL
            k_l (int): 保留脉冲数 kL
        """
        super().__init__()
        self.t_l = t_l
        self.k_l = k_l


    def extra_repr(self) -> str:
        return "t_l=%g, k_l=%d" % (self.t_l, self.k_l)


    def forward(self, t_volley: torch.Tensor, normalized: bool = True) -> torch.Tensor:
        """
        Args:
            t_volley (torch.Tensor): 时间簇 [B, M]
            normalized (bool): 是否为归一化时间簇
        Returns:
            out (torch.Tensor): 抑制后的时间簇
        """
        t_in = t_volley
        t_min = None
        if not normalized:
            t_in, t_min = _TV.normalize_time_volley(t_volley)
        out = lateral_inhibit(t_in, self.t_l, self.k_l)
        if not normalized and t_min is not None:
            present = _TV.is_spike_present(out)
            out = out + t_min.unsqueeze(-1)
            out = out.masked_fill(~present, _TV.EMPTY_SPIKE)
        return out
