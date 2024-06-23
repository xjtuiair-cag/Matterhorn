# -*- coding: UTF-8 -*-
"""
ANN转SNN小工具，完成ANN与SNN的转换。
"""


import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import matterhorn_pytorch.snn as snn
from typing import Iterable, Callable
from copy import deepcopy
from rich.progress import track


def ann_to_snn(model: nn.Module, demo_data: torch.utils.data.Dataset, mode: str = "max") -> snn.Module:
    """
    ANN转SNN。
    Args:
        model (nn.Module): 待转的ANN模型
        demo_data (torch.utils.data.Dataset): 示例数据，用于确定ReLU的最大值
        mode (str): 如何确定转换倍率lambda
    Returns:
        res (snn.Module): 转换后的SNN模型
    """
    model = deepcopy(model)
    res: snn.Module = None
    # 1. 逐个替换模块
    def _replace(ann_module: nn.Module) -> snn.Module:
        snn_module: snn.Module = None
        # 全连接层
        if isinstance(ann_module, nn.Linear):
            snn_module = snn.Linear(
                in_features = ann_module.in_features,
                out_features = ann_module.out_features,
                bias = ann_module.bias is not None,
                multi_time_step = False,
                device = ann_module.weight.device,
                dtype = ann_module.weight.dtype
            )
            params = ann_module.state_dict()
            for name in params:
                params[name] = params[name].clone().detach()
            snn_module.load_state_dict()
        # 卷积层
        elif isinstance(ann_module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
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
                multi_time_step = False,
                device = ann_module.weight.device,
                dtype = ann_module.weight.dtype
            )
            if isinstance(ann_module, nn.Conv1d):
                snn_module = snn.Conv1d(**kwargs)
            if isinstance(ann_module, nn.Conv2d):
                snn_module = snn.Conv2d(**kwargs)
            if isinstance(ann_module, nn.Conv3d):
                snn_module = snn.Conv3d(**kwargs)
            params = ann_module.state_dict()
            for name in params:
                params[name] = params[name].clone().detach()
            snn_module.load_state_dict(params)
        # 转置卷积层
        elif isinstance(ann_module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
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
                multi_time_step = False,
                device = ann_module.weight.device,
                dtype = ann_module.weight.dtype
            )
            if isinstance(ann_module, nn.ConvTranspose1d):
                snn_module = snn.ConvTranspose1d(**kwargs)
            if isinstance(ann_module, nn.ConvTranspose2d):
                snn_module = snn.ConvTranspose2d(**kwargs)
            if isinstance(ann_module, nn.ConvTranspose3d):
                snn_module = snn.ConvTranspose3d(**kwargs)
            params = ann_module.state_dict()
            for name in params:
                params[name] = params[name].clone().detach()
            snn_module.load_state_dict(params)
        elif isinstance(ann_module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            snn_module = snn.synapse.NormPlaceholder(
                num_features = ann_module.num_features,
                eps = ann_module.eps,
                momentum = ann_module.momentum,
                affine = ann_module.affine,
                track_running_stats = ann_module.track_running_stats,
                multi_time_step = False,
                device = ann_module.running_mean.device,
                dtype = ann_module.running_mean.dtype
            )
            params = ann_module.state_dict()
            for name in params:
                params[name] = params[name].clone().detach()
            snn_module.load_state_dict(params)
        # 激活函数
        elif isinstance(ann_module, (nn.ReLU, nn.LeakyReLU)):
            snn_module = snn.IF(
                u_threshold = 1.0,
                u_rest = 0.0,
                hard_reset = False,
                multi_time_step = False,
                reset_after_process = False
            )
            setattr(ann_module, "snn_module", snn_module)
        elif isinstance(ann_module, nn.Sequential):
            modules = []
            for module in ann_module:
                modules.append(_replace(module))
            snn_module = snn.Spatial(*modules)
        else:
            hybrid_module = deepcopy(ann_module)
            params = ann_module.state_dict()
            for name in params:
                params[name] = params[name].clone().detach()
            hybrid_module.load_state_dict(params)
            for name, module in ann_module.named_children():
                setattr(hybrid_module, name, _replace(module))
            snn_module = snn.Agent(
                nn_module = hybrid_module,
                force_spike_output = False,
                multi_time_step = False,
                reset_after_process = False
            )
        return snn_module
    
    res = snn.Temporal(
        module = _replace(model),
        reset_after_process = True
    )

    if mode is not None:
        global ann2snn_output_module_list
        ann2snn_output_module_list = []
        def register_hook(model: snn.Module, hook: Callable) -> Iterable:
            hooks = []
            hooks.append(model.register_forward_hook(hook))
            for module in model.children():
                hooks = hooks + register_hook(module, hook)
            return hooks
        
        def remove_hook(hooks: Iterable) -> None:
            for hook in hooks:
                hook.remove()

        # TODO: 2. 赋予前向钩子，记录放缩值，记录在每个IF神经元的lambda_l参数中
        def lambda_hook(model: snn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            if isinstance(model, (nn.ReLU, nn.LeakyReLU)):
                max_out = torch.max(output)
                snn_model: snn.Module = getattr(model, "snn_module")
                lambda_l = getattr(snn_model, "lambda_l") if hasattr(snn_model, "lambda_l") else torch.full_like(max_out, 1e-9)
                lambda_l = max(max_out, lambda_l)
                setattr(snn_model, "lambda_l", lambda_l)

        hooks = register_hook(model, lambda_hook)
        model.eval()
        with torch.no_grad():
            for x, y in track(demo_data, description = "Updating lambdas"):
                o = model(x[None])
        remove_hook(hooks)

        # 3. 根据所记录的lambda值，更新权重与偏置
        def scale_hook(model: snn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            global ann2snn_output_module_list
            from_module: snn.Module = None
            for out, module in reversed(ann2snn_output_module_list):
                if out is input[0]:
                    from_module = module
                    break
            ann2snn_output_module_list.append((output, model))
            
            is_synapse = lambda x: isinstance(x, (snn.Linear, snn.Conv1d, snn.Conv2d, snn.Conv3d, snn.ConvTranspose1d, snn.ConvTranspose2d, snn.ConvTranspose3d))
            is_norm = lambda x: isinstance(x, (snn.synapse.NormPlaceholder,))
            is_soma = lambda x: isinstance(x, (snn.IF,))

            if is_norm(model):
                synapse: snn.Module = None
                if is_synapse(from_module):
                    synapse = from_module
                if synapse is not None:
                    # BN -> Conv
                    norm_params = model.state_dict()
                    synapse_params = synapse.state_dict()
                    mu = norm_params["running_mean"]
                    sigma = (norm_params["running_var"] + getattr(model, "eps")) ** 0.5
                    gamma = norm_params["weight"]
                    beta = norm_params["bias"]
                    w = synapse_params["weight"].clone().detach()
                    b = synapse_params["bias"].clone().detach()
                    for i in range(len(mu)):
                        w[i] = gamma[i] / sigma[i] * w[i]
                        b[i] = gamma[i] / sigma[i] * (b[i] - mu[i]) + beta[i]
                    synapse_params["weight"] = w
                    synapse_params["bias"] = b
                    synapse.load_state_dict(synapse_params)
                    setattr(model, "synapse", synapse)
            elif is_soma(model):
                synapse: snn.Module = None
                if is_synapse(from_module):
                    synapse = from_module
                elif is_norm(from_module):
                    synapse = getattr(from_module, "synapse") if hasattr(from_module, "synapse") else None
                if synapse is not None:
                    # IF -> Conv
                    lambda_l = getattr(model, "lambda_l") if hasattr(model, "lambda_l") else 1.0
                    synapse_params = synapse.state_dict()
                    w = synapse_params["weight"].clone().detach()
                    b = synapse_params["bias"].clone().detach()
                    w = w / lambda_l
                    b = b / lambda_l
                    synapse_params["weight"] = w
                    synapse_params["bias"] = b
                    synapse.load_state_dict(synapse_params)
                    pass
            elif is_synapse(model):
                soma: snn.Module = None
                if is_soma(from_module):
                    soma = from_module
                if soma is not None:
                    # Conv -> IF
                    lambda_l = getattr(soma, "lambda_l") if hasattr(soma, "lambda_l") else 1.0
                    synapse_params = model.state_dict()
                    w = synapse_params["weight"].clone().detach()
                    w = w * lambda_l
                    synapse_params["weight"] = w
                    model.load_state_dict(synapse_params)
            return output
        
        hooks = register_hook(res, scale_hook)
        res.eval()
        with torch.no_grad():
            x, y = demo_data[0]
            o = res(x[None][None])
        remove_hook(hooks)
    
    def _remove_norm(before: snn.Module) -> snn.Module:
        after: snn.Module = None
        after = deepcopy(before)
        for name, module in before.named_children():
            if isinstance(module, snn.synapse.NormPlaceholder):
                setattr(after, name, snn.Identity(multi_time_step = False))
            elif isinstance(module, snn.IF):
                if hasattr(module, "lambda_l"):
                    delattr(module, "lambda_l")
            else:
                setattr(after, name, _remove_norm(module))
        return after
    res = _remove_norm(res)
    return res