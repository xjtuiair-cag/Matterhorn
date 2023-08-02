import torch
import torch.nn as nn


"""
脉冲神经网络的容器，用来容纳时间和空间维度的脉冲神经网络集合
建议先在空间维度上构建完整的脉冲神经网络结构，再在多个时间步之内进行模拟
"""


class Spatial(nn.Sequential):
    def __init__(self, *args):
        """
        SNN的空间容器
        用法同nn.Sequential，加入一些特殊的作用于SNN的函数
        @params:
            *args: [nn.Module] 按空间顺序传入的各个模块
        """
        super().__init__(*args)
    

    def n_reset(self):
        """
        一次性重置该序列中所有的神经元
        """
        for module in self:
            if hasattr(module, "n_reset"):
                module.n_reset()


class Temporal(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        """
        SNN的时间容器
        在多个时间步之内执行脉冲神经网络
        @params:
            model: nn.Module 所用来执行的单步模型
        """
        super().__init__()
        self.model = model
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，默认接受的张量形状为[T,B,...]（需要将时间维度通过permute等函数转到最外）
        @params:
            x: torch.Tensor 输入张量
        @return:
            y: torch.Tensor 输出张量
        """
        time_steps = x.shape[0]
        result = []
        for t in range(time_steps):
            result.append(self.model(x[t]))
        y = torch.stack(result)
        return y
    