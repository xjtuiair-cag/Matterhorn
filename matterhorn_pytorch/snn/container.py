# -*- coding: UTF-8 -*-
"""
脉冲神经网络的容器，用来容纳时间和空间维度的脉冲神经网络集合。
建议先在空间维度上构建完整的脉冲神经网络结构，再在多个时间步之内进行模拟。
"""


import torch
import torch.nn as nn
import matterhorn_pytorch.snn.functional as _SF
from matterhorn_pytorch.snn.skeleton import Module as _Module
from typing import Any as _Any, Iterable as _Iterable, Mapping as _Mapping, Tuple as _Tuple, Union as _Union, Optional as _Optional


def _safe_multi_step_mode_(module: _Module, if_on: bool, recursive: bool = True) -> _Module:
    if if_on and not module.multi_step_mode:
        if module.supports_multi_step_mode:
            module.multi_step_mode_(recursive = recursive)
        else:
            module = Temporal(module)
    if not if_on and module.multi_step_mode:
        if module.supports_single_step_mode:
            module.single_step_mode_(recursive = recursive)
        else:
            raise ValueError("Unsupported step mode conversion on module %s" % (module.__class__.__name__,))
    return module


class Container(_Module):
    def __init__(self) -> None:
        """
        容器的基类。
        """
        super().__init__()


class Sequential(Container, nn.Sequential):
    def __init__(self, *args: _Tuple[nn.Module]) -> None:
        """
        对Sequential进行重写，涵盖ANN与SNN的网络。
        Args:
            *args (*nn.Module): 按空间顺序传入的各个模块
            multi_step_mode (nn.Module): 是否支持多时间步模式。
        """
        multi_step_mode = False
        for module in args:
            if isinstance(module, _Module):
                multi_step_mode = multi_step_mode or module.multi_step_mode
        Container.__init__(self)
        nn.Sequential.__init__(self, *args)
        self.multi_step_mode_(multi_step_mode)


    def multi_step_mode_(self, if_on: bool = True, recursive: bool = True) -> nn.Module:
        """
        调整模型至多时间步模式。
        Args:
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
            recursive (bool): 是否递归调整子模块的时间步模式
        """
        Container.multi_step_mode_(self, if_on, recursive = False)
        for idx, module in enumerate(self):
            is_snn_module = isinstance(module, _Module)
            if is_snn_module:
                self[idx] = _safe_multi_step_mode_(module, if_on, recursive = recursive)
        return self


    def add_module(self, name: str, module: nn.Module) -> None:
        """
        将模块加入PyTorch的属性中。
        Args:
            name (str): 模块名
            module (nn.Module): 模块
        """
        if module is not None and isinstance(module, _Module):
            module.multi_step_mode_(self.multi_step_mode)
        return super().add_module(name, module)


    def __setitem__(self, idx: int, module: nn.Module) -> None:
        """
        更改模块。
        Args:
            idx (int): 模块索引
            module (nn.Module): 模块
        """
        if isinstance(module, _Module):
            module.multi_step_mode_(self.multi_step_mode)
        super().__setitem__(idx, module)


    def __add__(self, other: nn.Sequential) -> nn.Sequential:
        """
        拼接两个模块序列。
        Args:
            other (nn.Sequential): 另一个模块序列
        Returns:
            res (Sequential): 拼接完成的序列
        """
        multi_step_mode = self.multi_step_mode
        if isinstance(other, Sequential):
            multi_step_mode = multi_step_mode and other.multi_step_mode
        ret: nn.Sequential = super().__add__(other)
        res = Sequential()
        for module in ret:
            res.append(module)
        res = res.multi_step_mode_(multi_step_mode)
        return res


    def insert(self, index: int, module: nn.Module) -> nn.Sequential:
        """
        在某处插入一个模块。
        Args:
            idx (int): 模块索引
            module (nn.Module): 模块
        Returns:
            res (Sequential): 插入后的序列
        """
        if isinstance(module, _Module):
            module.multi_step_mode_(self.multi_step_mode)
        return super().insert(index, module)


    def forward(self, *args: _Tuple[torch.Tensor], **kwargs: _Mapping[str, _Any]) -> torch.Tensor:
        """
        前向传播函数，默认接受的张量形状为[B,...]（需要将时间维度通过permute等函数转到最外）
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            y (torch.Tensor): 输出张量
        """
        y = nn.Sequential.forward(self, *args, **kwargs)
        return y


class Spatial(Sequential):
    def __init__(self, *args: _Tuple[nn.Module]) -> None:
        """
        SNN的空间容器，用法同nn.Sequential，加入一些特殊的作用于SNN的函数。
        Args:
            *args (*nn.Module): 按空间顺序传入的各个模块
        """
        for module in args:
            self._check_if_snn_module(module)
        super().__init__(*args)
    

    def _check_if_snn_module(self, module: nn.Module):
        assert isinstance(module, _Module), "Module %s is not an SNN module." % (module.__class__.__name__)


    def add_module(self, name: str, module: _Module) -> None:
        """
        将模块加入PyTorch的属性中。
        Args:
            name (str): 模块名
            module (nn.Module): 模块
        """
        self._check_if_snn_module(module)
        return super().add_module(name, module)


    def __setitem__(self, idx: int, module: _Module) -> None:
        """
        更改模块。
        Args:
            idx (int): 模块索引
            module (nn.Module): 模块
        """
        self._check_if_snn_module(module)
        return super().__setitem__(idx, module)


    def __add__(self, other: "Spatial") -> "Spatial":
        """
        拼接两个模块序列。
        Args:
            other (nn.Sequential): 另一个模块序列
        Returns:
            res (Sequential): 拼接完成的序列
        """
        assert isinstance(other, Spatial), "Not a spatial module."
        ret = super().__add__(other)
        res = Spatial()
        for module in ret:
            res.append(module)
        return res


    def insert(self, index: int, module: nn.Module) -> nn.Sequential:
        """
        在某处插入一个模块。
        Args:
            idx (int): 模块索引
            module (nn.Module): 模块
        Returns:
            res (Sequential): 插入后的序列
        """
        self._check_if_snn_module(module)
        return super().insert(index, module)


class Temporal(Container):
    def __init__(self, module: _Module) -> None:
        """
        SNN的时间容器，在多个时间步之内执行脉冲神经网络。
        Args:
            module (nn.Module): 所用来执行的单步模型
        """
        assert isinstance(module, _Module) and not module.multi_step_mode, "Temporal container can only accept SNN module with single step mode."
        super().__init__()
        self.multi_step_mode_(recursive = False)
        self.module = module


    def forward_step(self, *args: _Tuple[torch.Tensor], **kwargs: _Mapping[str, _Any]) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            *args (*torch.Tensor): 输入
            **kwargs (str: Any): 输入
        Returns:
            res (torch.Tensor): 输出
        """
        if self.module.multi_step_mode:
            self.module.single_step_mode_()
        return self.module(*args, **kwargs)


class ModuleList(Container, nn.ModuleList):
    def __init__(self, modules: _Iterable[nn.Module] = None) -> None:
        """
        SNN的模块列表。
        Args:
            modules (nn.Module*): 加入模块列表中的模型
        """
        Container.__init__(self)
        nn.ModuleList.__init__(
            self,
            modules = modules
        )


    def multi_step_mode_(self, if_on: bool = True, recursive: bool = True) -> nn.Module:
        """
        调整模型至多时间步模式。
        Args:
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
            recursive (bool): 是否递归调整子模块的时间步模式
        """
        for idx, module in enumerate(self):
            is_snn_module = isinstance(module, _Module)
            if is_snn_module:
                self[idx] = _safe_multi_step_mode_(module, if_on, recursive = recursive)
        return self


    def __add__(self, other: _Iterable[nn.Module]) -> "ModuleList":
        """
        拼接模块列表。
        Args:
            other (nn.Module*): 另一个模块列表
        Returns:
            res (ModuleList): 拼接结果
        """
        ret = super().__add__(other)
        res = ModuleList()
        for i, module in enumerate(ret):
            res.add_module(str(i), module)
        return res


class ModuleDict(Container, nn.ModuleDict):
    def __init__(self, modules: _Mapping[str, nn.Module] = None) -> None:
        """
        SNN的模块字典。
        Args:
            modules ({str: nn.Module}): 加入模块字典中的模型
        """
        Container.__init__(self)
        nn.ModuleDict.__init__(
            self,
            modules = modules
        )


    def multi_step_mode_(self, if_on: bool = True, recursive: bool = True) -> nn.Module:
        """
        调整模型至多时间步模式。
        Args:
            if_on (bool): 当前需要调整为什么模式（True为多时间步模式，False为单时间步模式）
            recursive (bool): 是否递归调整子模块的时间步模式
        """
        for idx, module in self.items():
            is_snn_module = isinstance(module, _Module)
            if is_snn_module:
                self[idx] = _safe_multi_step_mode_(module, if_on, recursive = recursive)
        return self


class Agent(_Module):
    def __init__(self, nn_module: nn.Module, force_spike_output: bool = False) -> None:
        """
        ANN套壳，用于使ANN模块带有mth.snn.Module的方法
        Args:
            module (nn.Module): torch.nn模型
            force_spike_output (bool): 是否强制脉冲输出
        """
        super().__init__()
        assert not isinstance(nn_module, _Module), "The module %s.%s is already an SNN module." % (nn_module.__module__, nn_module.__class__.__name__)
        self.force_spike_output = force_spike_output
        self.nn_module: nn.Module = nn_module


    def __getattr__(self, name: str) -> _Any:
        """
        "."运算符重载，获取属性。
        Args:
            name (str): 属性名
        Returns:
            value (Any): 属性值
        """
        if name in ("nn_module",):
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except:
            return getattr(self.nn_module, name)


    def __setattr__(self, name: str, value: _Any) -> None:
        """
        ".="运算符重载，设置属性。
        Args:
            name (str): 属性名
            value (Any): 属性值
        """
        if name in ("nn_module",):
            modules = self.__dict__.get("_modules")
            modules["nn_module"] = value
            return
        if hasattr(self, name):
            return super().__setattr__(name, value)
        elif hasattr(self, "nn_module"):
            return setattr(self.nn_module, name, value)
        else:
            return super().__setattr__(name, value)


    def __delattr__(self, name: str) -> int:
        """
        "del ."运算符重载，删除属性。
        Args:
            name (str): 属性名
        """
        assert name != "nn_module", "Cannot remove the module and leave the agent empty."
        if hasattr(self, name):
            return super().__delattr__(name)
        elif hasattr(self, "nn_module"):
            return delattr(self.nn_module, name)
        else:
            raise AttributeError(f"Undefined attribute.")


    def __len__(self) -> int:
        """
        "len"运算符重载，获取长度。
        """
        return len(self.nn_module)


    def __getitem__(self, index: _Any) -> _Any:
        """
        "[]"运算符重载，获取键值。
        Args:
            index (Any): 键名
        Returns:
            value (Any): 键值
        """
        return self.nn_module[index]


    def __setitem__(self, index: _Any, value: _Any) -> None:
        """
        "[]="运算符重载，设置键值对。
        Args:
            index (Any): 键名
            value (Any): 键值
        """
        self.nn_module[index] = value


    def __delitem__(self, index: _Any) -> None:
        """
        "del []"运算符重载，删除键值对。
        Args:
            index (Any): 键名
        """
        del self.nn_module[index]


    def __iter__(self) -> _Any:
        """
        "iter"运算符重载，返回可迭代项的迭代器。
        """
        return iter(self.nn_module)


    def __next__(self) -> _Any:
        """
        "next"运算符重载，获取迭代器的下一个值。
        """
        return next(self.nn_module)


    def __add__(self, other: _Any) -> _Any:
        """
        "+"运算符重载，加法操作。
        Args:
            other (Any): 另一个加数
        Returns:
            res (Any): 加法结果
        """
        return self.nn_module + other


    def __iadd__(self, other: _Any) -> _Any:
        """
        "+="运算符重载，自加法操作。
        Args:
            other (Any): 另一个加数
        Returns:
            res (Any): 加法结果
        """
        return self.nn_module.__iadd__(other)
        

    def __mul__(self, other: _Any) -> _Any:
        """
        "*"运算符重载，乘法操作。
        Args:
            other (Any): 另一个被乘数
        Returns:
            res (Any): 乘法结果
        """
        return self.nn_module * other


    def __imul__(self, other: _Any) -> _Any:
        """
        "*="运算符重载，自乘法操作。
        Args:
            other (Any): 另一个被乘数
        Returns:
            res (Any): 乘法结果
        """
        return self.nn_module.__imul__(other)


    def __dir__(self) -> _Iterable:
        """
        "dir"运算符重载，返回所有属性和操作。
        Returns:
            attrs (str*): 属性和操作列表
        """
        return dir(self.nn_module)


    def reset(self) -> nn.Module:
        """
        重置模型。
        """
        super().reset()
        _ = [module.reset() if isinstance(module, _Module) else None for name, module in self.nn_module.named_children()] if self.nn_module is not None else None
        return self


    def detach(self) -> nn.Module:
        """
        将模型中的某些变量从其计算图中分离。
        """
        super().detach()
        _ = [module.detach() if isinstance(module, _Module) else None for name, module in self.nn_module.named_children()] if self.nn_module is not None else None
        return self


    def forward_step(self, *args: _Tuple[torch.Tensor], **kwargs: _Mapping[str, _Any]) -> torch.Tensor:
        """
        单个时间步的前向传播函数。
        Args:
            *args (*torch.Tensor): 输入
            **kwargs (str: Any): 输入
        Returns:
            res (torch.Tensor): 输出
        """
        res = self.nn_module(*args, **kwargs)
        return res


    def forward(self, *args: _Tuple[torch.Tensor], **kwargs: _Mapping[str, _Any]) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            *args (*torch.Tensor): 输入
            **kwargs (str: Any): 输入
        Returns:
            res (torch.Tensor): 输出
        """
        res = super().forward(*args, **kwargs)
        if self.force_spike_output:
            if isinstance(res, _Tuple):
                res = (_SF.to_spike_train(y) if isinstance(y, torch.Tensor) else y for y in res)
            else:
                res = _SF.to_spike_train(res)
        return res