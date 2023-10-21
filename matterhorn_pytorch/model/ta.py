# -*- coding: UTF-8 -*-
"""
逐元素注意力脉冲神经网络（Temporal-wise Attention Spiking Neural Networks），用于过滤事件稀疏的帧
Reference:
[Yao M, Gao H, Zhao G, et al. Temporal-wise attention spiking neural networks for event streams classification\[C\]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 10221-10230.]
(https://openaccess.thecvf.com/content/ICCV2021/html/Yao_Temporal-Wise_Attention_Spiking_Neural_Networks_for_Event_Streams_Classification_ICCV_2021_paper.html)
"""


import torch
import torch.nn as nn
import from matterhorn_pytorch.snn as snn
import math
try:
    from rich import print
except:
    pass


class TemporalWiseAttention(snn.Module):
    def __init__(self, time_steps: int, r: float, d_threshold: float) -> None:
        """
        Tempora-wise Attention连接层
        @params:
            time_steps: int 时间步长
            r: float 用于控制权重矩阵的大小(T*(T/r)和(T/r)*T)
            d_threshold: float 注意阈值，用于阶跃函数
        """
        super().__init__(
            multi_time_step = True
        )
        self.time_steps = time_steps
        self.r = r
        self.d_threshold = d_threshold

        self.fc1 = snn.Linear(self.time_steps, math.floor(self.time_steps / self.r), bias = False)
        self.fc2 = snn.Linear(math.floor(self.time_steps / self.r), self.time_steps, bias = False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.heaviside = snn.Rectangular()


    def reset(self) -> None:
        """
        重置模型。
        """
        self.fc1.reset()
        self.fc2.reset()
        self.heaviside.reset()
    

    def f_train(self, s: torch.Tensor) -> torch.Tensor:
        """
        训练模式下的前向传播，公式为$d^{n-1}=σ(W_{2}^{n}δ(W_{1}^{n}s^{n-1}]))$
        @params:
            s: torch.Tensor 统计向量$s^{n-1}$
        @return:
            d: torch.Tensor 分数向量$d^{n-1}$
        """
        dim = len(s.shape)
        if dim > 1:
            s = s.permute(1, 0) # 将形状[T, B]翻转为[B, T]
        d = self.sigmoid(self.fc2(self.relu(self.fc1(s))))
        if dim > 1:
            d = d.permute(1, 0) # 将形状[B, T]翻转为[T, B]
        return d


    def f_inf(self, s: torch.Tensor) -> torch.Tensor:
        """
        推理模式下的前向传播，公式为$d^{n-1}=σ(W_{2}^{n}δ(W_{1}^{n}s^{n-1}]))$
        @params:
            s: torch.Tensor 统计向量$s^{n-1}$，形状为[B, T]
        @return:
            d: torch.Tensor 分数向量$d^{n-1}$，形状为[B, T]
        """
        dim = len(s.shape)
        if dim > 1:
            s = s.permute(1, 0) # 将形状[T, B]翻转为[B, T]
        d = self.heaviside(self.sigmoid(self.fc2(self.relu(self.fc1(s)))) - self.d_threshold)
        if dim > 1:
            d = d.permute(1, 0) # 将形状[B, T]翻转为[T, B]
        return d


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        @params:
            x: torch.Tensor 原始脉冲张量
        @return:
            x: torch.Tensor 过滤后的脉冲张量
        """
        is_train = x.requires_grad
        dim = len(x.shape)
        s = x.clone().detach().requires_grad_(is_train)
        pre_permute = []
        post_permute = []
        # 获取统计向量$s^{n-1}$
        if dim > 2: # 前两维为[B, T]
            post_permute.append(dim - 2)
            post_permute.append(dim - 1)
            for i in range(2, dim):
                pre_permute.append(i)
                post_permute.append(i - 2)
                s = torch.mean(s, dim = 2)
            pre_permute.append(0)
            pre_permute.append(1)
        # 获取分数向量$d^{n-1}$：$d^{n-1}=TA(s^{n-1})$
        if is_train:
            d = self.f_train(s)
        else:
            d = self.f_inf(s)
        # 过滤：$X^{t,n-1}=d_{t}^{n-1}X^{t,n-1}$
        if dim > 2:
            x = x.permute(*pre_permute)
            x = d * x
            x = x.permute(*post_permute)
        else:
            x = d * x
        return x