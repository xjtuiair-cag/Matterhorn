# -*- coding: UTF-8 -*-
"""
脉冲神经网络的编码机制。
注意：此单元可能会改变张量形状。
"""


import torch
import torch.nn as nn
from matterhorn.snn.skeleton import Module
try:
    from rich import print
except:
    pass


class Encoder(Module):
    def __init__(self) -> None:
        """
        编码器的基类。编码器是一个多时间步模型。
        """
        super().__init__(
            multi_time_step = True
        )


    def supports_single_time_step(self) -> bool:
        """
        是否支持单个时间步。
        @return:
            if_support: bool 是否支持单个时间步
        """
        return False


    def supports_multi_time_step(self) -> bool:
        """
        是否支持多个时间步。
        @return:
            if_support: bool 是否支持多个时间步
        """
        return True


class Direct(Encoder):
    def __init__(self) -> None:
        """
        直接编码，直接对传入的脉冲（事件）数据进行编码
        """
        super().__init__()
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        直接编码的前向传播函数，直接将第二个维度作为时间维度，转置到首个维度上
        @params:
            x: torch.Tensor 输入张量，形状为[B,T,...]
        @return:
            y: torch.Tensor 输出张量，形状为[T,B,...]
        """
        idx = [i for i, j in enumerate(x.shape)]
        assert len(idx) >= 2, "There is no time steps."
        idx[0], idx[1] = idx[1], idx[0]
        y = x.permute(*idx)
        return y


class Poisson(Encoder):
    def __init__(self, time_steps: int = 1, max_value: float = 1.0, min_value: float = 0.0) -> None:
        """
        泊松编码（速率编码），将值转化为脉冲发放率（多步）
        @params:
            time_steps: int 生成的时间步长
            max_value: float 最大值
            min_value: float 最小值
        """
        super().__init__()
        assert max_value > min_value, "Max value is less than min value."
        self.time_steps = time_steps
        self.max_value = max_value
        self.min_value = min_value


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return "time_steps=%d, max=%.3f, min=%.3f" % (self.time_steps, self.max_value, self.min_value)


    def forward_single(self, x:torch.Tensor) -> torch.Tensor:
        """
        单步前向传播函数。
        @params:
            x: torch.Tensor 输入张量，形状为[B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[B,...]
        """
        r = torch.rand_like(x)
        y = r.le(x).to(x)
        return y
    

    def forward_multiple(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        """
        多步前向传播函数。
        @params:
            x: torch.Tensor 输入张量，形状为[B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[T, B,...]
        """
        y_seq = []
        for t in range(time_steps):
            y_seq.append(self.forward_single(x))
        y = torch.stack(y_seq)
        return y


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        泊松编码的前向传播函数，将值$V$转化为该时间步$t$内的脉冲$O^{0}(t)$
        @params:
            x: torch.Tensor 输入张量，形状为[B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[T,B,...]
        """
        x = (x - self.min_value) / (self.max_value - self.min_value)
        if self.time_steps <= 1:
            y = self.forward_single(x)
        else:
            y = self.forward_multiple(x, self.time_steps)
        return y


class Temporal(Encoder):
    def __init__(self, time_steps: int = 1, max_value: float = 1.0, min_value: float = 0.0, prob: float = 0.75, reset_after_process: bool = True) -> None:
        """
        时间编码，在某个时间之前不会产生脉冲，在某个时间之后随机产生脉冲
        @params:
            time_steps: int 生成的时间步长
            max_value: float 最大值
            min_value: float 最小值
            prob: float 若达到了时间，以多大的概率发放脉冲，范围为[0, 1]
            reset_after_process: bool 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__()
        assert max_value > min_value, "Max value is less than min value."
        self.time_steps = time_steps
        self.max_value = max_value
        self.min_value = min_value
        self.prob = prob
        self.reset_after_process = reset_after_process
        self.reset()


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        @return:
            repr_str: str 参数表
        """
        return "time_steps=%d, max=%.3f, min=%.3f, spike_freq=%.3f" % (self.time_steps, self.max_value, self.min_value, self.prob)


    def reset(self) -> None:
        """
        重置编码器
        """
        self.current_time_step = 0


    def forward_single(self, x:torch.Tensor) -> torch.Tensor:
        """
        单步前向传播函数。
        @params:
            x: torch.Tensor 输入张量，形状为[B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[B,...]
        """
        f = (self.current_time_step + 0.0 - x).ge(0.0).to(x)
        r = torch.rand_like(x).le(self.prob).to(x)
        y = f * r
        self.current_time_step += 1
        return y
    

    def forward_multiple(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        """
        多步前向传播函数。
        @params:
            x: torch.Tensor 输入张量，形状为[B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[T, B,...]
        """
        y_seq = []
        for t in range(time_steps):
            y_seq.append(self.forward_single(x))
        y = torch.stack(y_seq)
        return y


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        泊松编码的前向传播函数，将值$V$转化为该时间步$t$内的脉冲$O^{0}(t)$
        @params:
            x: torch.Tensor 输入张量，形状为[B,...]
        @return:
            y: torch.Tensor 输出张量，形状为[T,B,...]
        """
        x = (x - self.min_value) / (self.max_value - self.min_value) * self.time_steps
        if self.time_steps <= 1:
            y = self.forward_single(x)
        else:
            y = self.forward_multiple(x, self.time_steps)
        if self.reset_after_process:
            self.reset()
        return y