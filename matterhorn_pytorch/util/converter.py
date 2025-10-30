# -*- coding: UTF-8 -*-
"""
ANN转SNN小工具，完成ANN与SNN的转换。
"""


import torch
import torch.nn as nn
import matterhorn_pytorch.snn as snn
from typing import Tuple, Iterable, Callable, Optional
from torch.utils.data import Dataset
from copy import deepcopy
from rich import print
from rich.progress import track


_NN_MODULE_LINEAR = (nn.Linear,)
_NN_MODULE_CONV = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
_NN_MODULE_ICONV = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
_NN_MODULE_NORM = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)
_NN_MODULE_ACTIVATIONS = (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU, nn.GELU, nn.SiLU)
_NN_MODULE_MAX_POOL = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)
_NN_MODULE_AVG_POOL = (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)
_NN_MODULE_RESHAPE = (nn.Flatten, nn.Unflatten)
_NN_MODULE_DROPOUT = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)


def ann_to_snn(model: nn.Module, demo_data: Dataset, pre_process: Callable = lambda x: x[0], replace_rule: Optional[Callable] = None, mode: str = "max") -> snn.Module:
    """
    ANN转SNN。
    Args:
        model (nn.Module): 待转的ANN模型
        demo_data (torch.utils.data.Dataset): 示例数据，用于确定ReLU的最大值
        pre_process (Callable): 数据预处理方法
        replace_rule (Callable): 自定义结点替换规则，否则按照默认节点替换规则
        mode (str): 如何确定转换倍率lambda
    Returns:
        res (snn.Module): 转换后的SNN模型
    """
    global act_sn_lambdas
    act_sn_lambdas = {}

    model = deepcopy(model)
    res: snn.Module = None
    # 1. 逐个替换模块
    def _clone_params(ann_module: nn.Module, snn_module: snn.Module) -> snn.Module:
        global act_sn_lambdas
        params = ann_module.state_dict()
        new_params = snn_module.state_dict()
        for name in params:
            new_params[name] = params[name].clone()
        snn_module.load_state_dict(new_params)
        return snn_module

    def _replace(ann_module: nn.Module) -> snn.Module:
        global act_sn_lambdas
        snn_module: snn.Module = None
        if replace_rule is not None:
            snn_module = replace_rule(deepcopy(ann_module))
            if snn_module is not None:
                return snn_module
        if isinstance(ann_module, snn.Module):
            snn_module = deepcopy(ann_module)
        # 全连接层
        elif isinstance(ann_module, _NN_MODULE_LINEAR):
            snn_module = snn.Linear(
                in_features = ann_module.in_features,
                out_features = ann_module.out_features,
                bias = ann_module.bias is not None,
                device = ann_module.weight.device,
                dtype = ann_module.weight.dtype
            )
            snn_module = _clone_params(ann_module, snn_module)
        # 卷积层
        elif isinstance(ann_module, _NN_MODULE_CONV):
            kwargs = dict(
                in_channels = ann_module.in_channels,
                out_channels = ann_module.out_channels,
                kernel_size = ann_module.kernel_size,
                stride = ann_module.stride,
                padding = ann_module.padding,
                dilation = ann_module.dilation,
                groups = ann_module.groups,
                bias = ann_module.bias is not None,
                padding_mode = ann_module.padding_mode,
                device = ann_module.weight.device,
                dtype = ann_module.weight.dtype
            )
            if isinstance(ann_module, nn.Conv1d):
                snn_module = snn.Conv1d(**kwargs)
            elif isinstance(ann_module, nn.Conv2d):
                snn_module = snn.Conv2d(**kwargs)
            elif isinstance(ann_module, nn.Conv3d):
                snn_module = snn.Conv3d(**kwargs)
            snn_module = _clone_params(ann_module, snn_module)
        # 转置卷积层
        elif isinstance(ann_module, _NN_MODULE_ICONV):
            kwargs = dict(
                in_channels = ann_module.in_channels,
                out_channels = ann_module.out_channels,
                kernel_size = ann_module.kernel_size,
                stride = ann_module.stride,
                padding = ann_module.padding,
                output_padding = ann_module.output_padding,
                groups = ann_module.groups,
                bias = ann_module.bias is not None,
                dilation = ann_module.dilation,
                padding_mode = ann_module.padding_mode,
                device = ann_module.weight.device,
                dtype = ann_module.weight.dtype
            )
            if isinstance(ann_module, nn.ConvTranspose1d):
                snn_module = snn.ConvTranspose1d(**kwargs)
            elif isinstance(ann_module, nn.ConvTranspose2d):
                snn_module = snn.ConvTranspose2d(**kwargs)
            elif isinstance(ann_module, nn.ConvTranspose3d):
                snn_module = snn.ConvTranspose3d(**kwargs)
            snn_module = _clone_params(ann_module, snn_module)
        elif isinstance(ann_module, _NN_MODULE_NORM):
            kwargs = dict(
                num_features = ann_module.num_features,
                eps = ann_module.eps,
                momentum = ann_module.momentum,
                affine = ann_module.affine,
                track_running_stats = ann_module.track_running_stats,
                device = ann_module.running_mean.device,
                dtype = ann_module.running_mean.dtype
            )
            if isinstance(ann_module, nn.BatchNorm1d):
                snn_module = snn.BatchNorm1d(**kwargs)
            elif isinstance(ann_module, nn.BatchNorm2d):
                snn_module = snn.BatchNorm2d(**kwargs)
            elif isinstance(ann_module, nn.BatchNorm3d):
                snn_module = snn.BatchNorm3d(**kwargs)
            snn_module = _clone_params(ann_module, snn_module)
        # 激活函数
        elif isinstance(ann_module, _NN_MODULE_ACTIVATIONS):
            snn_module = snn.IF(
                u_threshold = 1.0,
                u_rest = 0.0,
                hard_reset = False,
                return_states = False
            )
            act_sn_lambdas[id(snn_module)] = {
                "module": snn_module,
                "act_idx": id(ann_module),
                "lambda_l": 1e-9
            }
        # 最大池化
        elif isinstance(ann_module, _NN_MODULE_MAX_POOL):
            kwargs = dict(
                kernel_size = ann_module.kernel_size,
                stride = ann_module.stride,
                padding = ann_module.padding,
                dilation = ann_module.dilation,
                return_indices = ann_module.return_indices,
                ceil_mode = ann_module.ceil_mode
            )
            if isinstance(ann_module, nn.MaxPool1d):
                snn_module = snn.MaxPool1d(**kwargs)
            elif isinstance(ann_module, nn.MaxPool2d):
                snn_module = snn.MaxPool2d(**kwargs)
            elif isinstance(ann_module, nn.MaxPool3d):
                snn_module = snn.MaxPool3d(**kwargs)
        # 平均池化
        elif isinstance(ann_module, _NN_MODULE_AVG_POOL):
            kwargs = dict(
                kernel_size = ann_module.kernel_size,
                stride = ann_module.stride,
                padding = ann_module.padding,
                ceil_mode = ann_module.ceil_mode,
                count_include_pad = ann_module.count_include_pad
            )
            if isinstance(ann_module, nn.MaxPool1d):
                snn_module = snn.AvgPool1d(**kwargs)
            elif isinstance(ann_module, nn.MaxPool2d):
                kwargs["divisor_override"] = ann_module.divisor_override
                snn_module = snn.AvgPool2d(**kwargs)
            elif isinstance(ann_module, nn.MaxPool3d):
                kwargs["divisor_override"] = ann_module.divisor_override
                snn_module = snn.AvgPool3d(**kwargs)
        # 展开和反展开
        elif isinstance(ann_module, _NN_MODULE_RESHAPE):
            if isinstance(ann_module, nn.Flatten):
                snn_module = snn.Flatten(
                    start_dim = ann_module.start_dim,
                    end_dim = ann_module.end_dim
                )
            elif isinstance(ann_module, nn.Unflatten):
                snn_module = snn.Unflatten(
                    dim = ann_module.dim,
                    unflattened_size = ann_module.unflattened_size
                )
        # 遗忘层
        elif isinstance(ann_module, _NN_MODULE_DROPOUT):
            kwargs = dict(
                p = ann_module.p,
                inplace = ann_module.inplace
            )
            if isinstance(ann_module, nn.Dropout1d):
                snn_module = snn.Dropout1d(**kwargs)
            elif isinstance(ann_module, nn.Dropout2d):
                snn_module = snn.Dropout2d(**kwargs)
            elif isinstance(ann_module, nn.Dropout3d):
                snn_module = snn.Dropout3d(**kwargs)
            else:
                snn_module = snn.Dropout(**kwargs)
        # 顺序执行
        elif isinstance(ann_module, nn.Sequential):
            modules = []
            for module in ann_module:
                modules.append(_replace(module))
            snn_module = snn.Sequential(*modules, return_states = False)
        # 模块集合
        elif isinstance(ann_module, (nn.ModuleList, nn.ModuleDict)):
            if isinstance(ann_module, nn.ModuleList):
                modules = list()
                for module in ann_module:
                    modules.append(_replace(module))
                snn_module = nn.ModuleList(
                    modules = modules
                )
            elif isinstance(ann_module, nn.ModuleDict):
                modules = dict()
                for name in ann_module:
                    module = ann_module[name]
                    modules[name] = _replace(module)
                snn_module = nn.ModuleDict(
                    modules = modules
                )
        # 其它自定义ANN模块
        else:
            snn_module = deepcopy(ann_module)
            for name, module in snn_module.named_children():
                sub_module = _replace(module)
                setattr(snn_module, name, sub_module)
            snn_module = _clone_params(ann_module, snn_module)
        return snn_module
    
    res = _replace(model)

    if mode is not None:
        global graph_cache
        graph_cache = []
        def register_hook(model: snn.Module, hook: Callable) -> Iterable:
            global act_sn_lambdas
            hooks = []
            hooks.append(model.register_forward_hook(hook))
            for name, module in model.named_children():
                hooks = hooks + register_hook(module, hook)
            return hooks
        
        def remove_hook(hooks: Iterable) -> None:
            global act_sn_lambdas
            for hook in hooks:
                hook.remove()

        # 2. 赋予前向钩子，记录放缩值，记录在每个IF神经元的lambda_l参数中
        def lambda_hook(model: snn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            global act_sn_lambdas
            if isinstance(model, _NN_MODULE_ACTIVATIONS):
                max_out = torch.max(output)
                for sn_idx in act_sn_lambdas:
                    if id(model) == act_sn_lambdas[sn_idx]["act_idx"]:
                        lambda_l = act_sn_lambdas[sn_idx]["lambda_l"]
                        lambda_l = max(max_out, lambda_l)
                        act_sn_lambdas[sn_idx]["lambda_l"] = lambda_l

        hooks = register_hook(model, lambda_hook)
        model.eval()
        with torch.no_grad():
            for item in track(demo_data, description = "Updating lambdas"):
                x = pre_process(item)
                if x is None:
                    continue
                o = model(x[None])
        remove_hook(hooks)

        # 3. 根据所记录的lambda值，更新权重与偏置
        def scale_hook(model: snn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            global act_sn_lambdas
            is_node = (not len(list(model.children())) and not isinstance(model, snn.firing.Firing)) or isinstance(model, snn.soma.Soma)
            if not is_node:
                return
            if isinstance(output, Tuple):
                for o in output:
                    for i in input:
                        if o is not i:
                            graph_cache.append((model, i, o))
            else:
                for i in input:
                    if output is not i:
                        graph_cache.append((model, i, output))
            
            def _search_node(start_tensor: torch.Tensor, return_cond: Callable, halt_cond: Callable) -> Tuple[snn.Module, bool, int]:
                global graph_cache, act_sn_lambdas
                return_modules: Iterable[snn.Module] = []
                search_succeed: bool = False
                for _model, _input, _output in reversed(graph_cache):
                    if _output is start_tensor:
                        if return_cond(_model):
                            return_modules = [_model]
                            search_succeed = True
                        elif halt_cond(_model):
                            return_modules = []
                            search_succeed = False
                        else:
                            modules, succeed = _search_node(_input, return_cond, halt_cond)
                            if succeed:
                                return_modules = return_modules + modules
                                search_succeed = True
                return return_modules, search_succeed
            
            is_synapse = lambda x: isinstance(x, (snn.Linear, snn.Conv1d, snn.Conv2d, snn.Conv3d, snn.ConvTranspose1d, snn.ConvTranspose2d, snn.ConvTranspose3d))
            is_norm = lambda x: isinstance(x, (snn.BatchNorm1d, snn.BatchNorm2d, snn.BatchNorm3d))
            is_soma = lambda x: isinstance(x, (snn.IF,))

            if is_norm(model):
                synapses, _ = _search_node(output, lambda x: is_synapse(x), lambda x: not is_synapse(x) and x is not model)
                for synapse in synapses:
                    # BN -> Conv
                    # print("BatchNorm\n", model, "\n", synapse, "\n")
                    norm_params = model.state_dict()
                    synapse_params = synapse.state_dict()
                    mu = norm_params["running_mean"]
                    sigma = (norm_params["running_var"] + getattr(model, "eps")) ** 0.5
                    gamma = norm_params["weight"]
                    beta = norm_params["bias"]
                    w = synapse_params["weight"].clone()
                    for i, el in enumerate(mu):
                        w[i] = gamma[i] / sigma[i] * w[i]
                    synapse_params["weight"] = w
                    has_bias = "bias" in synapse_params
                    if has_bias:
                        b = synapse_params["bias"].clone()
                        b = gamma / sigma * (b - mu) + beta
                        synapse_params["bias"] = b
                    synapse.load_state_dict(synapse_params)
            elif is_soma(model):
                synapses, _ = _search_node(output, lambda x: is_synapse(x), lambda x: is_soma(x) and x is not model)
                for synapse in synapses:
                    # IF -> Conv
                    # print("IF -> Conv\n", model, "\n", synapse, "\n")
                    lambda_l = act_sn_lambdas[id(model)]["lambda_l"]
                    synapse_params = synapse.state_dict()
                    w = synapse_params["weight"].clone()
                    w = w / lambda_l
                    synapse_params["weight"] = w
                    has_bias = "bias" in synapse_params
                    if has_bias:
                        b = synapse_params["bias"].clone()
                        b = b / lambda_l
                        synapse_params["bias"] = b
                    synapse.load_state_dict(synapse_params)
            elif is_synapse(model):
                somas, _ = _search_node(output, lambda x: is_soma(x), lambda x: is_synapse(x) and x is not model)
                for soma in somas:
                    # Conv -> IF
                    # print("Conv -> IF\n", model, "\n", soma, "\n")
                    lambda_l = act_sn_lambdas[id(soma)]["lambda_l"]
                    synapse_params = model.state_dict()
                    w = synapse_params["weight"].clone()
                    w = w * lambda_l
                    synapse_params["weight"] = w
                    model.load_state_dict(synapse_params)
            return
        
        hooks = register_hook(res, scale_hook)
        res.eval()
        with torch.no_grad():
            x = pre_process(demo_data[0])
            if x is not None:
                o = res(x[None][None])
        remove_hook(hooks)
    
    def _remove_norm(before: snn.Module) -> snn.Module:
        global act_sn_lambdas
        after: snn.Module = None
        after = deepcopy(before)
        for name, module in before.named_children():
            if isinstance(module, (snn.BatchNorm1d, snn.BatchNorm2d, snn.BatchNorm3d)):
                setattr(after, name, nn.Identity())
            else:
                setattr(after, name, _remove_norm(module))
        return after
    
    res = _remove_norm(res)
    del act_sn_lambdas
    return res