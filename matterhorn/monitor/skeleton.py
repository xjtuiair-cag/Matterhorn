import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from typing import Iterable, Callable, Optional, Union


class Monitor(nn.Module):
    def __init__(self, group: nn.ModuleList, variables: Optional[Iterable] = None, record_at: Optional[Union[Iterable, Callable]] = None) -> None:
        """
        监视器的框架
        @params:
            group: nn.ModuleList 该监视器属于哪一个组别的
            module: nn.Module 该监视器所监视的模块
            variables: List | None 需要监视哪些变量，为None就是全部监视。可选"i"（输入），"o"（输出），"x"（突触后电位）或者"u"（胞体电位）
            record_at: List | Callable | None: 什么时候应该截取。为List时，当匹配到对应时间时截取数据；为Callable时，输入current_time_step和x，当返回True时截取数据；为None时，截取每一时刻的数据。
        """
        super().__init__()
        self.group = group
        self.group.append(self)
        self.variables = variables
        self.record_time_steps = record_at


    def reset(self) -> None:
        """
        重置模型。
        """


    def record_at(self, time_steps: Optional[Union[Iterable, Callable]]) -> None:
        """
        设置截取数据的时间。
        @params:
            time_steps: List | Callable | None: 什么时候应该截取。为List时，当匹配到对应时间时截取数据；为Callable时，输入current_time_step和x，当返回True时截取数据；为None时，截取每一时刻的数据。
        """
        self.record_time_steps = time_steps


    def if_record(self, t: int, key: str) -> bool:
        """
        判断当前时间步是否需要记录
        @params:
            t: int 当前时间步
            key: str 当前数据的标签
            value: torch.Tensor 当前数据
        @return:
            if_record: bool 是否需要记录
        """
    
    
    def record(self, t: int, key: str, value: torch.Tensor) -> None:
        """
        截取数据
        @params:
            t: int 当前时间步
            key: str 当前数据的标签
            value: torch.Tensor 当前数据
        """
        pass


    def show(self) -> None:
        """
        展示图像
        """
        pass


    def forward(self, t: int, key: str, value: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，记录后返回原值
        @params:
            t: int 当前时间步
            key: str 当前数据的标签
            value: torch.Tensor 当前数据
        @return:
            value: torch.Tensor 当前数据
        """
        if not self.if_record(t, key):
            return value
        self.record(t, key, value.detach())
        return value
        