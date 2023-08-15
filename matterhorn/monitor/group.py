import torch
import torch.nn as nn
from typing import Iterable, Optional


class MonitorGroup(nn.ModuleList):
    def __init__(self, modules: Optional[Iterable[nn.Module]] = None) -> None:
        super().__init__(modules)


    def reset(self) -> None:
        for module in self:
            module.reset()

    def record_at(self, time_steps: Iterable) -> None:
        """
        设置截取数据的时间。
        @params:
            time_steps: List | Callable | None: 什么时候应该截取。为List时，当匹配到对应时间时截取数据；为Callable时，输入current_time_step和x，当返回True时截取数据；为None时，截取每一时刻的数据。
        """
        for module in self:
            module.record_at(time_steps)


    def show(self) -> None:
        for module in self:
            module.show()