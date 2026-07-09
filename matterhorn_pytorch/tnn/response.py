# -*- coding: UTF-8 -*-
"""
兴奋性突触复合响应函数 E_i(t, w_i)。

基于 Smith 书第6.6节 Model II 的分段线性近似（表6.1、图6.14）。
"""


import torch
from typing import Optional as _Optional


def compound_response(
    t_rel: torch.Tensor,
    weight: torch.Tensor,
    tr: float,
    tf: float,
    gamma: float
) -> torch.Tensor:
    """
    计算分段线性复合兴奋性响应 E_i(t, w_i)。
    Args:
        t_rel (torch.Tensor): 相对时间（自输入脉冲起算），任意形状
        weight (torch.Tensor): 突触权重 w_i，可与 t_rel 广播
        tr (float): 上升时间 Tr
        tf (float): 下降时间 Tf
        gamma (float): 缩放因子 γ = Tmax/Wmax
    Returns:
        response (torch.Tensor): 响应幅值，与 t_rel 同形
    """
    w = weight.clamp(min = 0.0)
    t_peak = gamma * (w - 1.0) + tr
    t_end = t_peak + tf
    t_peak = torch.clamp(t_peak, min = 1e-6)

    rise = w * torch.clamp(t_rel, min = 0.0) / t_peak
    fall = w * (t_end - t_rel) / tf
    resp = torch.where(
        t_rel < 0.0,
        torch.zeros_like(t_rel),
        torch.where(
            t_rel <= t_peak,
            rise,
            torch.where(t_rel <= t_end, fall, torch.zeros_like(t_rel))
        )
    )
    resp = resp * (w > 0.0).to(resp.dtype)
    return resp


def membrane_potential(
    times: torch.Tensor,
    spike_times: torch.Tensor,
    weights: torch.Tensor,
    tr: float,
    tf: float,
    gamma: float,
    connectivity: _Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算膜电位轨迹 P(t) = Σ_i E_i(t - t_i, w_i)。
    Args:
        times (torch.Tensor): 采样时刻，形状 [T]
        spike_times (torch.Tensor): 输入脉冲时间，形状 [B, I]
        weights (torch.Tensor): 突触权重，形状 [B, I] 或 [I]
        tr (float): Tr
        tf (float): Tf
        gamma (float): γ
        connectivity (torch.Tensor | None): 连接掩码 [I] 或 [B, I]，0 表示断开
    Returns:
        potential (torch.Tensor): P(t)，形状 [B, T]
    """
    batch, num_inputs = spike_times.shape
    present = torch.isfinite(spike_times)
    w = weights
    if w.ndim == 1:
        w = w.unsqueeze(0).expand(batch, -1)
    if connectivity is not None:
        mask = connectivity.to(w.dtype)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        w = w * mask

    t_grid = times.view(1, 1, -1)
    t_in = spike_times.unsqueeze(-1)
    t_rel = t_grid - t_in
    resp = compound_response(t_rel, w.unsqueeze(-1), tr, tf, gamma)
    resp = resp * present.unsqueeze(-1).to(resp.dtype)
    return resp.sum(dim = 1)
