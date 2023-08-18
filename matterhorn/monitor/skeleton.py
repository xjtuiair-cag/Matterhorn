import torch
import torch.nn as nn
from typing import Iterable


class Monitor(nn.Module):
    """
    监视器有两种记录方式：
    （1）作为模块插入网络中，记录所经过的脉冲或电位；
    （2）加入各个神经元中，被动地接受信息（张量）并记录。
    记录使用record函数，展示图片使用show函数，导出数据使用export函数。
    """


    def __init__(self, name: str, group: nn.ModuleList, key: str = "s") -> None:
        """
        监视器的框架。
        @params:
            name: str 该监视器的名字
            group: nn.ModuleList 该监视器属于哪一个组别的
        """
        super().__init__()
        self.name = name
        self.group = group
        self.group.append(self)
        self.key = key
        self.reset


    def reset(self) -> None:
        """
        重置模型。
        """
        self.records = {}


    def record(self, key: str, value: torch.Tensor) -> None:
        """
        通过各个模块主动向监视器中记录数据。
        @params:
            t: int 当前时间步
            key: str 当前数据的标签
            value: torch.Tensor 当前数据
        """
        if key not in self.records:
            self.records[key] = []
        self.records[key].append(value.detach().cpu())


    def export(self, key: str) -> torch.Tensor:
        """
        输入对应的key，导出所需要的数据。
        @params:
            key: str 要导出哪一组数据
        @return:
            x: torch.Tensor 时间维度，t轴
            y: torch.Tensor n+1维张量，第一维是时间，其余维与输入的张量相同
        """
        if key not in self.records:
            return None
        x = torch.arange(0, len(self.records[key]))
        y = torch.stack(self.records[key])
        return x, y


    def show(self, keys: Iterable[str] = None) -> None:
        """
        展示图像。
        @params:
            keys: str 可选，要展示哪些数据，None为全部展示
        """
        pass


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，可以放在两个module之间记录中间值，记录后返回原值。
        @params:
            value: torch.Tensor 当前数据
        @return:
            value: torch.Tensor 当前数据
        """
        self.record(self.key, value)
        return value