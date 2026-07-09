# -*- coding: UTF-8 -*-
"""
TNN 训练工具：基于输出脉冲时间的监督学习，以及 STDP 无监督学习。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.tnn.cognitive import CognitiveColumn
from matterhorn_pytorch.tnn.column import ExcitatoryColumn
from matterhorn_pytorch.tnn.neuron import ExcitatoryNeuron
from matterhorn_pytorch.tnn import stdp as _TS
from matterhorn_pytorch.tnn import volley as _TV
from typing import Iterable as _Iterable, Optional as _Optional


def spike_time_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fill: float = 1e6
) -> torch.Tensor:
    """
    输出脉冲时间的均方误差损失（无脉冲位置用大罚值替代）。
    Args:
        pred (torch.Tensor): 预测时间 [B] 或 [B, N]
        target (torch.Tensor): 目标时间
        fill (float): 无脉冲时的替代值
    Returns:
        loss (torch.Tensor): 标量损失
    """
    p = pred.masked_fill(~torch.isfinite(pred), fill)
    t = target.masked_fill(~torch.isfinite(target), fill)
    return ((p - t) ** 2).mean()


def train_excitatory_neuron_stdp(
    neuron: ExcitatoryNeuron,
    training_volleys: _Iterable[torch.Tensor],
    learning_rate: float = 0.05,
    epochs: int = 1
) -> ExcitatoryNeuron:
    """
    使用 volley STDP 无监督训练单个兴奋性神经元。
    Args:
        neuron (ExcitatoryNeuron): 待训练神经元
        training_volleys (Iterable[torch.Tensor]): 训练时间簇序列 [B, I]
        learning_rate (float): 学习率
        epochs (int): 遍历训练集次数
    Returns:
        neuron (ExcitatoryNeuron): 训练后的神经元（原地修改权重）
    """
    neuron.train()
    for _ in range(epochs):
        for volley in training_volleys:
            t_norm, _ = _TV.normalize_time_volley(volley)
            out, _ = neuron(volley, global_times = True)
            for b in range(volley.shape[0]):
                t_out = out[b]
                if not torch.isfinite(t_out):
                    continue
                _, t_ref = _TV.normalize_time_volley(volley[b:b + 1])
                row = t_norm[b]
                new_w = _TS.volley_stdp_update(
                    neuron.weight.data,
                    row.unsqueeze(0),
                    (t_out - t_ref).squeeze(),
                    learning_rate = learning_rate,
                    t_max = neuron.t_max,
                    w_max = neuron.w_max,
                    connectivity = neuron.connectivity
                )
                neuron.weight.data.copy_(new_w)
    return neuron


def train_column_mean_weights(
    column: ExcitatoryColumn,
    training_volleys: torch.Tensor
) -> ExcitatoryColumn:
    """
    使用 Model II 均值法为兴奋柱中每个神经元设置权重（Smith 书第6.6.2节）。
    Args:
        column (ExcitatoryColumn): 兴奋柱
        training_volleys (torch.Tensor): 训练时间簇 [S, I]
    Returns:
        column (ExcitatoryColumn): 训练后的兴奋柱
    """
    for n, neuron in enumerate(column.neurons):
        mask = column.connectivity[n].bool()
        mean_t = torch.full((neuron.num_inputs,), _TV.EMPTY_SPIKE, device = training_volleys.device, dtype = training_volleys.dtype)
        for i in range(neuron.num_inputs):
            if not mask[i]:
                continue
            col = training_volleys[:, i]
            present = torch.isfinite(col)
            if present.any():
                mean_t[i] = col[present].mean()
        neuron.weight.data.copy_(
            _TS.mean_time_to_weight(mean_t, neuron.t_max, neuron.gamma) * column.connectivity[n]
        )
    return column


class TNNTrainer:
    """
    TNN 监督训练器：对含 ExcitatoryNeuron / Column / CognitiveColumn 的模型做 BPTT 风格优化。
    """

    def __init__(self, model: nn.Module, learning_rate: float = 1e-2) -> None:
        """
        Args:
            model (nn.Module): 待训练 TNN 模型
            learning_rate (float): Adam 学习率
        """
        self.model = model
        self.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr = learning_rate
        )


    def supervised_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        单步监督训练。
        Args:
            inputs (torch.Tensor): 输入时间簇 [B, I]
            targets (torch.Tensor): 目标输出时间 [B] 或 [B, N]
        Returns:
            loss_value (float): 损失值
        """
        self.model.train()
        self.optimizer.zero_grad()
        if isinstance(self.model, CognitiveColumn):
            outputs, _ = self.model(merged_input = inputs)
        elif isinstance(self.model, ExcitatoryColumn):
            outputs, _ = self.model(inputs)
        elif isinstance(self.model, ExcitatoryNeuron):
            outputs, _ = self.model(inputs)
        else:
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
        loss = spike_time_mse_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
