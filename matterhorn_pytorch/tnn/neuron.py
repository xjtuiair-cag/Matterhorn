# -*- coding: UTF-8 -*-
"""
兴奋性神经元（Excitatory Neuron Model II）。

基于 Smith 书第6.6节：分段线性复合响应 + 膜电位求和 + 阈值交叉确定输出脉冲时间。
支持通过替代梯度进行反向传播训练。
"""


import torch
import torch.nn as nn
import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.skeleton import Module as _Module
from matterhorn_pytorch.tnn import response as _TR
from matterhorn_pytorch.tnn import volley as _TV
from typing import Any as _Any, Tuple as _Tuple, Optional as _Optional


class _excitatory_neuron_eval(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _Any,
        spike_times: torch.Tensor,
        weights: torch.Tensor,
        theta: torch.Tensor,
        tr: torch.Tensor,
        tf: torch.Tensor,
        gamma: torch.Tensor,
        t_max: torch.Tensor,
        num_steps: int,
        connectivity: _Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        兴奋性神经元评估前向传播。
        Args:
            spike_times (torch.Tensor): 归一化输入脉冲时间 [B, I]
            weights (torch.Tensor): 突触权重 [I]
            theta (torch.Tensor): 发放阈值 θ
            tr, tf, gamma, t_max (torch.Tensor): 标量参数张量
            num_steps (int): 膜电位离散采样步数
            connectivity (torch.Tensor | None): 连接掩码 [I]
        Returns:
            out_time (torch.Tensor): 输出脉冲时间 [B]，无脉冲为 inf
        """
        device = spike_times.device
        dtype = spike_times.dtype
        batch = spike_times.shape[0]
        times = torch.linspace(0.0, float(t_max.item()), num_steps, device = device, dtype = dtype)
        potential = _TR.membrane_potential(
            times, spike_times, weights, float(tr.item()), float(tf.item()), float(gamma.item()), connectivity
        )
        crossed = potential >= theta
        has_cross = crossed.any(dim = -1)
        first_idx = crossed.to(dtype).argmax(dim = -1)
        out_time = times[first_idx]
        out_time = out_time.masked_fill(~has_cross, _TV.EMPTY_SPIKE)

        ctx.save_for_backward(spike_times, weights, theta, tr, tf, gamma, t_max, times, potential, out_time)
        ctx.num_steps = num_steps
        ctx.connectivity = connectivity
        return out_time


    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor) -> _Tuple:
        spike_times, weights, theta, tr, tf, gamma, t_max, times, potential, out_time = ctx.saved_tensors
        batch, num_inputs = spike_times.shape
        grad_spike = torch.zeros_like(spike_times)
        grad_weight = torch.zeros_like(weights)
        grad_theta = torch.zeros_like(theta)
        grad_tr = grad_tf = grad_gamma = grad_tmax = None
        grad_steps = None
        grad_conn = None

        present_out = torch.isfinite(out_time)
        if not present_out.any():
            return (grad_spike, grad_weight, grad_theta, grad_tr, grad_tf, grad_gamma, grad_tmax, grad_steps, grad_conn)

        for b in range(batch):
            if not present_out[b]:
                continue
            g = grad_output[b]
            t_out = out_time[b]
            idx = (times - t_out).abs().argmin()
            p = potential[b, idx]
            margin = p - theta
            surrogate = _SF.bp_gaussian(margin.unsqueeze(0), 0.5).squeeze(0) * g

            t_s = times[idx]
            w = weights
            if ctx.connectivity is not None:
                w = w * ctx.connectivity.to(w.dtype)
            present = torch.isfinite(spike_times[b])
            t_rel = t_s - spike_times[b]
            d_resp_dw = _TR.compound_response(
                t_rel.unsqueeze(-1),
                w.unsqueeze(-1),
                float(tr.item()), float(tf.item()), float(gamma.item())
            ).squeeze(-1)
            grad_weight = grad_weight + surrogate * d_resp_dw * present.to(d_resp_dw.dtype)

            for i in range(num_inputs):
                if not present[i]:
                    continue
                eps = 1e-3
                t_rel_hi = (t_s + eps) - spike_times[b, i]
                t_rel_lo = (t_s - eps) - spike_times[b, i]
                r_hi = _TR.compound_response(
                    t_rel_hi.unsqueeze(0), w[i].unsqueeze(0),
                    float(tr.item()), float(tf.item()), float(gamma.item())
                ).squeeze()
                r_lo = _TR.compound_response(
                    t_rel_lo.unsqueeze(0), w[i].unsqueeze(0),
                    float(tr.item()), float(tf.item()), float(gamma.item())
                ).squeeze()
                grad_spike[b, i] = surrogate * (r_lo - r_hi) / (2.0 * eps)

        return (grad_spike, grad_weight, grad_theta, grad_tr, grad_tf, grad_gamma, grad_tmax, grad_steps, grad_conn)


class ExcitatoryNeuron(_Module):
    """
    兴奋性神经元（Model II）。
    输入为各突触首次脉冲时间戳（时间簇），输出为神经元首次发放时间。
    """

    def __init__(
        self,
        num_inputs: int,
        w_max: float = 4.0,
        t_max: float = 16.0,
        tr: float = 10.0,
        tf: float = 40.0,
        theta: float = 19.0,
        num_steps: int = 256,
        normalize_input: bool = True,
        connectivity: _Optional[torch.Tensor] = None,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        """
        Args:
            num_inputs (int): 输入线数 M
            w_max (float): 最大权重 Wmax
            t_max (float): 时间窗 Tmax
            tr (float): 响应上升时间 Tr
            tf (float): 响应下降时间 Tf
            theta (float): 膜电位阈值 θ
            num_steps (int): 膜电位轨迹离散采样点数
            normalize_input (bool): 是否对输入簇做局部归一化
            connectivity (torch.Tensor | None): 输入连接掩码 [M]
            device (torch.device): 计算设备
            dtype (torch.dtype): 数据类型
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.w_max = w_max
        self.t_max = t_max
        self.tr = tr
        self.tf = tf
        self.theta = theta
        self.num_steps = num_steps
        self.normalize_input = normalize_input
        self.gamma = t_max / w_max

        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.zeros(num_inputs, **factory))
        if connectivity is None:
            connectivity = torch.ones(num_inputs, **factory)
        self.register_buffer("connectivity", connectivity.to(**factory))
        self.register_buffer("_theta_buf", torch.tensor(theta, **factory))
        self.register_buffer("_tr_buf", torch.tensor(tr, **factory))
        self.register_buffer("_tf_buf", torch.tensor(tf, **factory))
        self.register_buffer("_gamma_buf", torch.tensor(self.gamma, **factory))
        self.register_buffer("_tmax_buf", torch.tensor(t_max, **factory))


    def extra_repr(self) -> str:
        return ", ".join([
            "num_inputs=%d" % self.num_inputs,
            "w_max=%g" % self.w_max,
            "t_max=%g" % self.t_max,
            "theta=%g" % self.theta
        ])


    def forward(self, spike_times: torch.Tensor, global_times: bool = True) -> _Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：输入时间簇 → 输出脉冲时间。
        Args:
            spike_times (torch.Tensor): 输入脉冲时间戳 [B, M]，无脉冲为 inf
            global_times (bool): 输入/输出是否为全局时间；True 时内部做归一化/反归一化
        Returns:
            out_time (torch.Tensor): 输出脉冲全局时间 [B]
            t_min (torch.Tensor): 本次输入簇的最早脉冲时间 [B]（供下游使用）
        """
        t_in = spike_times
        t_min = torch.zeros(spike_times.shape[0], device = spike_times.device, dtype = spike_times.dtype)
        if self.normalize_input and global_times:
            t_in, t_min = _TV.normalize_time_volley(spike_times)

        if self.training and self.weight.requires_grad:
            out_norm = _excitatory_neuron_eval.apply(
                t_in, self.weight, self._theta_buf, self._tr_buf, self._tf_buf,
                self._gamma_buf, self._tmax_buf, self.num_steps, self.connectivity
            )
        else:
            device, dtype = spike_times.device, spike_times.dtype
            times = torch.linspace(0.0, self.t_max, self.num_steps, device = device, dtype = dtype)
            potential = _TR.membrane_potential(
                times, t_in, self.weight, self.tr, self.tf, self.gamma, self.connectivity
            )
            crossed = potential >= self.theta
            has_cross = crossed.any(dim = -1)
            first_idx = crossed.to(dtype).argmax(dim = -1)
            out_norm = times[first_idx]
            out_norm = out_norm.masked_fill(~has_cross, _TV.EMPTY_SPIKE)

        if self.normalize_input and global_times:
            present = torch.isfinite(out_norm)
            out_time = out_norm + t_min
            out_time = out_time.masked_fill(~present, _TV.EMPTY_SPIKE)
        else:
            out_time = out_norm
        return out_time, t_min
