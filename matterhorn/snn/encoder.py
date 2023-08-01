import torch
import torch.nn as nn


class Poisson(nn.Module):
    def __init__(self, max_value: float = 1.0, min_value: float = 0.0) -> None:
        """
        泊松编码（速率编码），将指转化为脉冲发放率
        @params:
            max_value: 最大值
        """
        super().__init__()
        assert max_value > min_value, "Max value is less than min value."
        self.max_value = max_value
        self.min_value = min_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        泊松编码的前向传播函数，将值$V$转化为该时间步$t$内的脉冲$O^{0}(t)$
        """
        x = (x - self.min_value) / (self.max_value - self.min_value)
        r = torch.rand_like(x)
        o = r.le(x).to(x)
        return o


class Temporal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        