# -*- coding: UTF-8 -*-
"""
TNN 脉冲时序依赖可塑性（STDP）。

实现 Smith 书第6.4.3节的分段线性更新函数 G（图6.6），
适用于以首次脉冲时间戳表示的脉冲簇（volley）训练。
"""


import torch
from typing import Tuple as _Tuple, Optional as _Optional


def stdp_gate(delta_t: torch.Tensor, t_max: float, slope: float = 0.1) -> torch.Tensor:
    """
    分段线性 STDP 门控函数 G(Δt)，Δt = t_in - t_out。
    输入先于输出（Δt < 0）→ 增强（+1）；输入后于输出（Δt > 0）→ 减弱（-1）。
    Args:
        delta_t (torch.Tensor): 时间差
        t_max (float): 编码窗 Tmax，用于远端饱和
        slope (float): Δt=0 附近的线性斜率
    Returns:
        gate (torch.Tensor): 更新方向与幅度
    """
    pos = torch.clamp(1.0 + slope * delta_t / t_max, min = -1.0, max = 1.0)
    neg = torch.clamp(-1.0 + slope * delta_t / t_max, min = -1.0, max = 1.0)
    gate = torch.where(delta_t < 0.0, torch.ones_like(delta_t), torch.where(delta_t > 0.0, -torch.ones_like(delta_t), torch.zeros_like(delta_t)))
    near_zero = delta_t.abs() < (t_max * 0.05)
    gate = torch.where(delta_t < 0.0, torch.where(near_zero, pos, gate), gate)
    gate = torch.where(delta_t > 0.0, torch.where(near_zero, neg, gate), gate)
    return gate


def volley_stdp_update(
    weights: torch.Tensor,
    input_times: torch.Tensor,
    output_time: torch.Tensor,
    learning_rate: float = 0.05,
    t_max: float = 16.0,
    w_max: float = 4.0,
    connectivity: _Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    基于脉冲簇的局部 STDP 权重更新（Smith 书第6.4.3节）。
    Args:
        weights (torch.Tensor): 突触权重 [I] 或 [O, I]
        input_times (torch.Tensor): 输入脉冲时间 [B, I] 或 [I]
        output_time (torch.Tensor): 输出脉冲时间标量或 [B]
        learning_rate (float): 学习率 λ
        t_max (float): Tmax
        w_max (float): Wmax
        connectivity (torch.Tensor | None): 连接掩码
    Returns:
        new_weights (torch.Tensor): 更新后的权重
    """
    w = weights.clone()
    if input_times.ndim == 1:
        input_times = input_times.unsqueeze(0)
    if output_time.ndim == 0:
        output_time = output_time.unsqueeze(0)

    batch = input_times.shape[0]
    for b in range(batch):
        t_out = output_time[b]
        if not torch.isfinite(t_out):
            continue
        row = input_times[b]
        present = torch.isfinite(row)
        delta = row - t_out
        gate = stdp_gate(delta, t_max)
        gate = gate * present.to(gate.dtype)
        if w.ndim == 1:
            w = w + learning_rate * gate
        else:
            w[b] = w[b] + learning_rate * gate

    if connectivity is not None:
        w = w * connectivity.to(w.dtype)
    return w.clamp(0.0, w_max)


def mean_time_to_weight(
    mean_times: torch.Tensor,
    t_max: float,
    gamma: float
) -> torch.Tensor:
    """
    Model II 非 STDP 训练：wi = (Tmax - ti_mean) / γ（Smith 书第6.6.2节）。
    Args:
        mean_times (torch.Tensor): 各输入线平均脉冲时间
        t_max (float): Tmax
        gamma (float): γ = Tmax/Wmax
    Returns:
        weights (torch.Tensor): 训练后权重
    """
    present = torch.isfinite(mean_times)
    w = (t_max - mean_times) / gamma
    w = w.masked_fill(~present, 0.0)
    return w.clamp(min = 0.0)
