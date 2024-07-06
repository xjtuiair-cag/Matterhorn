# -*- coding: UTF-8 -*-
"""
脉冲神经网络的容器，用来容纳时间和空间维度的脉冲神经网络集合。
建议先在空间维度上构建完整的脉冲神经网络结构，再在多个时间步之内进行模拟。
"""


import torch
import torch.nn as nn
import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.skeleton import Module as _Module
from typing import Iterable as _Iterable, Mapping as _Mapping, Tuple as _Tuple


class Container(_Module):
    def __init__(self, multi_time_step: bool = False, reset_after_process: bool = False) -> None:
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
        Container.__init__(
            self,
            multi_time_step = False
        )
        nn.Sequential.__init__(self, *args)
        for module in self:
            is_snn_module = isinstance(module, _Module)
            if is_snn_module:
                assert self.supports_single_time_step, "Cannot put a multi time step model into spatial container."
                self.multi_time_step_(False)


    @property
    def supports_multi_time_step(self) -> bool:
        """
        是否支持多个时间步。
        Returns:
            if_support (bool): 是否支持多个时间步
        """
        return False


class Temporal(Container):
    def __init__(self, module: nn.Module, reset_after_process: bool = False) -> None:
        """
        SNN的时间容器，在多个时间步之内执行脉冲神经网络。
        Args:
            module (nn.Module): 所用来执行的单步模型
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
        """
        is_snn_module = isinstance(module, _Module)
        if is_snn_module:
            assert module.multi_time_step == False, "You cannot put a multi-time-step module %s into temporal container" % (module.__class__.__name__,)
        super().__init__(
            multi_time_step = True,
            reset_after_process = reset_after_process
        )
        self.module = module


    def extra_repr(self) -> str:
        """
        额外的表达式，把参数之类的放进来。
        Returns:
            repr_str (str): 参数表
        """
        return "reset_after_process=%s" % (str(self.reset_after_process),)


    @property
    def supports_single_time_step(self) -> bool:
        """
        是否支持单个时间步。
        Returns:
            if_support (bool): 是否支持单个时间步
        """
        return False


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播函数，默认接受的张量形状为[T,B,...]（需要将时间维度通过permute等函数转到最外）
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            y (torch.Tensor): 输出张量
        """
        if not isinstance(args, _Tuple):
            args = (args,)
        time_steps = args[0].shape[0]
        result = []
        for t in range(time_steps):
            args_t = (x[t] if isinstance(x, torch.Tensor) else x for x in args)
            kwargs_t = {k: x[t] if isinstance(x, torch.Tensor) else x for k, x in kwargs}
            result.append(self.module(*args_t, **kwargs_t))
        if isinstance(result[0], _Tuple):
            y = (torch.stack([result[t][col] for t in range(len(result))]) if isinstance(result[0][col], torch.Tensor) else result[0][col] for col in range(len(result[0])))
        else:
            y = torch.stack(result)
        return y


class Sequential(Container, nn.Sequential):
    def __init__(self, *args, reset_after_process: bool = False) -> None:
        """
        对Sequential进行重写，涵盖ANN与SNN的网络。
        Args:
            args (*nn.Module): 按空间顺序传入的各个模块
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
        """
        multi_time_step = True
        # all_snn_module_single_time_step = True
        # for module in args:
        #     is_snn_module = isinstance(module, _Module)
        #     if is_snn_module:
        #         all_snn_module_single_time_step = all_snn_module_single_time_step and not module.multi_time_step
        #     else:
        #         all_snn_module_single_time_step = False
        # if all_snn_module_single_time_step:
        #     multi_time_step = False
        Container.__init__(
            self,
            multi_time_step = multi_time_step,
            reset_after_process = reset_after_process
        )
        nn.Sequential.__init__(self, *args)
        convert_indices = []
        remove_indices = []
        last_single_step_idx = -2
        last_snn_idx = -2
        # 遍历所有传入的模块，确保它们是ANN模块或是多时间步SNN模块。
        for module_idx in range(len(self)):
            module = self[module_idx]
            is_snn_module = isinstance(module, _Module)
            # 如果是SNN模块。
            if is_snn_module:
                is_multi_time_step = module.multi_time_step
                # 如果是单时间步SNN模块。
                if not is_multi_time_step:
                    # 如果可以将它转成多时间步SNN模块，那就直接转成多时间步SNN模块。
                    if module.supports_multi_time_step:
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


    @property
    def supports_single_time_step(self) -> bool:
        """
        是否支持单个时间步。
        Returns:
            if_support (bool): 是否支持单个时间步
        """
        return False


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播函数，默认接受的张量形状为[T,B,...]（需要将时间维度通过permute等函数转到最外）
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            y (torch.Tensor): 输出张量
        """
        y = nn.Sequential.forward(self, *args, **kwargs)
        return y


class ModuleList(Container, nn.ModuleList):
    def __init__(self, modules: _Iterable[nn.Module] = None) -> None:
        Container.__init__(self)
        nn.ModuleList.__init__(
            self,
            modules = modules
        )


class ModuleDict(Container, nn.ModuleDict):
    def __init__(self, modules: _Mapping[str, nn.Module] = None) -> None:
        Container.__init__(self)
        nn.ModuleDict.__init__(
            self,
            modules = modules
        )


class Agent(_Module):
    def __init__(self, nn_module: nn.Module, force_spike_output: bool = False, multi_time_step: bool = False, reset_after_process: bool = False) -> None:
        """
        ANN套壳，用于使ANN模块带有mth.snn.Module的方法
        Args:
            module (nn.Module): torch.nn模型
            force_spike_output (bool): 是否强制脉冲输出
            multi_time_step (bool): 是否调整为多个时间步模式
            reset_after_process (bool): 是否在执行完后自动重置，若为False则需要手动重置
        """
        is_snn_module = isinstance(nn_module, _Module)
        assert not is_snn_module, "Already an SNN module."
        super().__init__(
            multi_time_step = multi_time_step,
            reset_after_process = reset_after_process
        )
        self.nn_module: nn.Module = nn_module
        self.force_spike_output = force_spike_output


    def multi_time_step_(self, if_on: bool) -> nn.Module:
        """
        调整模型的多时间步模式。
        Args
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
        """
        super().multi_time_step_(if_on)
        for module in self.nn_module.children():
            is_snn_module = isinstance(module, _Module)
            if is_snn_module:
                module.multi_time_step_(if_on)
                assert module.multi_time_step == self.multi_time_step, "Unmatched step mode"
        return self


    def reset(self) -> nn.Module:
        """
        重置模型。
        """
        super().reset()
        for module in self.nn_module.children():
            is_snn_module = isinstance(module, _Module)
            if is_snn_module:
                module.reset()
        return self


    def detach(self) -> nn.Module:
        """
        将模型中的某些变量从其计算图中分离。
        """
        super().detach()
        for module in self.nn_module.children():
            is_snn_module = isinstance(module, _Module)
            if is_snn_module:
                module.detach()
        return self


    def forward_single_time_step(self, *args, **kwargs) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            *args: 输入
            **kwargs: 输入
        Returns:
            res (torch.Tensor): 输出
        """
        if not isinstance(args, _Tuple):
            args = (args,)
        res = self.nn_module(*args, **kwargs)
        return res


    def forward_multi_time_step(self, *args, **kwargs) -> torch.Tensor:
        """
        多个时间步的前向传播函数。
        Args:
            *args: 输入
            **kwargs: 输入
        Returns:
            res (torch.Tensor): 输出
        """
        args, kwargs, tb = _SF.merge_time_steps_batch_size(args, kwargs)
        res = self.forward_single_time_step(*args, **kwargs)
        res = _SF.split_time_steps_batch_size(res, tb)
        return res


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            *args: 输入
            **kwargs: 输入
        Returns:
            res (torch.Tensor): 输出
        """
        res = super().forward(*args, **kwargs)
        if self.force_spike_output:
            if isinstance(res, _Tuple):
                res = (_SF.val_to_spike(y) if isinstance(y, torch.Tensor) else y for y in res)
            else:
                res = _SF.val_to_spike(res)
        return res