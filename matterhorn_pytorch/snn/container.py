# -*- coding: UTF-8 -*-
"""
脉冲神经网络的容器，用来容纳时间和空间维度的脉冲神经网络集合。
建议先在空间维度上构建完整的脉冲神经网络结构，再在多个时间步之内进行模拟。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.snn.skeleton import Module
try:
    from rich import print
except:
    pass


class Container(Module):
    def __init__(self, multi_time_step: bool = False, reset_after_process: bool = True) -> None:
        super().__init__(
            multi_time_step = multi_time_step,
            reset_after_process = reset_after_process
        )


class Spatial(Container, nn.Sequential):
    def __init__(self, *args) -> None:
        """
        SNN的空间容器，用法同nn.Sequential，加入一些特殊的作用于SNN的函数。
        Args:
            *args ([nn.Module]): 按空间顺序传入的各个模块
        """
        Container.__init__(self)
        nn.Sequential.__init__(self, *args)
        self.multi_time_step__ = False
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                self.multi_time_step__ = self.multi_time_step__ or module.multi_time_step


    def multi_time_step_(self, if_on: bool) -> nn.Module:
        """
        调整模型的多时间步模式。
        Args
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
        """
        self.multi_time_step__ = False
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                res = module.multi_time_step_(if_on)
                self.multi_time_step__ = self.multi_time_step__ or module.multi_time_step
        return self


    def reset(self) -> None:
        """
        一次重置该序列中所有的神经元。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.reset()


    def start_step(self) -> None:
        """
        开始STDP训练。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.start_step()


    def stop_step(self) -> None:
        """
        停止STDP训练。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.stop_step()
    

    def step_once(self) -> None:
        """
        一次部署所有结点的STDP训练。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.step_once()


class Temporal(Container):
    def __init__(self, module: nn.Module, reset_after_process: bool = True) -> None:
        """
        SNN的时间容器，在多个时间步之内执行脉冲神经网络。
        Args:
            module (nn.Module): 所用来执行的单步模型
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
        """
        is_snn_module = isinstance(module, Module)
        if is_snn_module:
            assert module.multi_time_step == False, "You cannot put a multi time step module %s into temporal container" % (module.__class__.__name__,)
        super().__init__(
            multi_time_step = True,
            reset_after_process = reset_after_process
        )
        self.module = module
        self.step_after_process = False


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return "reset_after_process=%s" % (str(self.reset_after_process),)


    def supports_single_time_step(self) -> bool:
        """
        是否支持单个时间步。
        Returns:
            if_support (bool): 是否支持单个时间步
        """
        return False


    def reset(self) -> None:
        """
        重置模型。
        """
        is_snn_module = isinstance(self.module, Module)
        if is_snn_module:
            self.module.reset()

    
    def start_step(self) -> None:
        """
        开始STDP训练。
        """
        self.step_after_process = True
        is_snn_module = isinstance(self.module, Module)
        if is_snn_module:
            self.module.start_step()
    

    def stop_step(self) -> None:
        """
        停止STDP训练。
        """
        self.step_after_process = False
        is_snn_module = isinstance(self.module, Module)
        if is_snn_module:
            self.module.stop_step()
    

    def step_once(self) -> None:
        """
        部署结点的STDP训练。
        """
        is_snn_module = isinstance(self.module, Module)
        if is_snn_module:
            self.module.step_once()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，默认接受的张量形状为[T,B,...]（需要将时间维度通过permute等函数转到最外）
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            y (torch.Tensor): 输出张量
        """
        time_steps = x.shape[0]
        result = []
        for t in range(time_steps):
            result.append(self.module(x[t]))
        y = torch.stack(result)
        if self.step_after_process:
            self.step_once()
        if self.reset_after_process:
            self.reset()
        return y


class Sequential(Container, nn.Sequential):
    def __init__(self, *args, reset_after_process: bool = True) -> None:
        """
        对Sequential进行重写，涵盖ANN与SNN的网络。
        Args:
            *args ([nn.Module]): 按空间顺序传入的各个模块
        """
        multi_time_step = True
        # all_snn_module_single_time_step = True
        # for module in args:
        #     is_snn_module = isinstance(module, Module)
        #     if is_snn_module:
        #         all_snn_module_single_time_step = all_snn_module_single_time_step and not module.multi_time_step
        #     else:
        #         all_snn_module_single_time_step = False
        # if all_snn_module_single_time_step:
        #     multi_time_step = False
        Container.__init__(self, multi_time_step = multi_time_step)
        nn.Sequential.__init__(self, *args)
        convert_indices = []
        remove_indices = []
        last_single_step_idx = -2
        last_snn_idx = -2
        # 遍历所有传入的模块，确保它们是ANN模块或是多时间步SNN模块。
        for module_idx in range(len(self)):
            module = self[module_idx]
            is_snn_module = isinstance(module, Module)
            # 如果是SNN模块。
            if is_snn_module:
                is_multi_time_step = module.multi_time_step
                # 如果是单时间步SNN模块。
                if not is_multi_time_step:
                    # 如果可以将它转成多时间步SNN模块，那就直接转成多时间步SNN模块。
                    if module.supports_multi_time_step():
                        module.multi_time_step_(True)
                        self[module_idx] = module
                    # 否则，记住当前SNN模块的索引，在之后将其合并。
                    else:
                        # 如果它的前一个模块也是不能转成多时间步SNN模块的单时间步SNN模块，将其插入到前一个模块所在的模块列表中，并删除该模块。
                        if last_single_step_idx == module_idx - 1:
                            convert_indices[-1].append(module_idx)
                            remove_indices.append(module_idx)
                        # 否则，新建一组需要合并的模块列表。
                        else:
                            convert_indices.append([module_idx, module_idx])
                        last_single_step_idx = module_idx
                # 否则，是多时间步SNN模块，不用处理。
                else:
                    pass
                last_snn_idx = module_idx
            # 否则，是ANN模块，不用处理。
            else:
                pass
        # 将需要进行合并的模块进行合并。
        for indices in convert_indices:
            # 若一组里面只有一个单元需要合并，那就直接在其外面套上时间容器即可。
            if len(indices) == 2:
                self[indices[0]] = Temporal(
                    self[indices[1]],
                    reset_after_process = reset_after_process
                )
            # 否则，一组里面有很多单元需要合并，除了时间容器之外，还需要套上一层空间容器来让各个模块顺序执行。
            else:
                module_list = []
                for module_idx in indices[1:]:
                    module_list.append(self[module_idx])
                self[indices[0]] = Temporal(
                    Spatial(
                        *module_list
                    ),
                    reset_after_process = reset_after_process
                )
        # 将需要解除连接的模块解除连接。逆序是为了不会破坏原索引。
        remove_indices = remove_indices[::-1]
        for idx in remove_indices:
            del self[idx]


    def reset(self) -> None:
        """
        一次重置该序列中所有的神经元。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.reset()


    def start_step(self) -> None:
        """
        开始STDP训练。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.start_step()


    def stop_step(self) -> None:
        """
        停止STDP训练。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.stop_step()
    

    def step_once(self) -> None:
        """
        一次部署所有结点的STDP训练。
        """
        for module in self:
            is_snn_module = isinstance(module, Module)
            if is_snn_module:
                module.step_once()