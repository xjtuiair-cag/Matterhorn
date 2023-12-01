# -*- coding: UTF-8 -*-
"""
（未完工，测试功能）SNN模型的保存与提取。
"""


import torch
import torch.nn as nn
import numpy as np
import os
import json
import inspect
import difflib
from typing import List, Tuple, Iterator, Dict, Iterable


import matterhorn_pytorch


import __main__


basic_types = [int, float, str, bool]
seq_types = [List, Dict, Tuple, np.ndarray, torch.Tensor]


def load_model_from_dict(info: Dict) -> nn.Module:
    model_class: nn.Module
    model_class = eval(info["module_type"])
    args = []
    param_list = {}
    fixed_params = [s for s in info["fixed_params"]]
    sub_modules = [s for s in info["sub_modules"]]
    mixed_params = fixed_params + sub_modules
    get_most_fit = lambda list, name: list[np.argmax([difflib.SequenceMatcher(None, name, s).quick_ratio() for s in list])]
    init_params = inspect.signature(model_class).parameters
    for name in init_params:
        if name == "args":
            args = [load_model_from_dict(info["sub_modules"][s]) for s in sub_modules]
        param = init_params[name]
        param_type = param.annotation
        if param.default is not inspect.Parameter.empty:
            continue
        if param_type in basic_types and len(fixed_params):
            param_list[name] = info["fixed_params"][get_most_fit(fixed_params, name)]
        if param_type in seq_types and len(fixed_params):
            param_list[name] = info["fixed_params"][get_most_fit(fixed_params, name)]
        if param_type in [nn.Module, matterhorn_pytorch.snn.Module] and len(sub_modules):
            param_list[name] = load_model_from_dict(info["sub_modules"][get_most_fit(sub_modules, name)])
        if param_type in [inspect._empty] and len(mixed_params):
            temp_name = get_most_fit(mixed_params, name)
            if temp_name in fixed_params:
                param_list[name] = info["fixed_params"][temp_name]
            else:
                param_list[name] = load_model_from_dict(info["sub_modules"][temp_name])
    print(model_class, args, param_list)
    model = model_class(*args, **param_list)
    for param in fixed_params:
        setattr(model, param, info["fixed_params"][param])
    for module in sub_modules:
        setattr(model, module, load_model_from_dict(info["sub_modules"][module]))
    return model


def load_model_from_json(json_string: str) -> nn.Module:
    model_dict = json.loads(json_string)

    pass


def save_model_to_dict(model: nn.Module) -> Dict:
    info = {
        "module_type": model.__module__ + "." + model.__class__.__name__,
        "fixed_params": {},
        "sub_modules": {}
    }
    for name in model.__dict__:
        param = model.__dict__[name]
        param_type = type(param)
        if param_type in basic_types:
            info["fixed_params"][name] = param
        if param_type in seq_types:
            for idx, val in enumerate(param):
                print(idx, val)
                if not val in basic_types:
                    continue
            info["fixed_params"][name] = list(param)
    for name, module in model._modules.items():
        info["sub_modules"][name] = save_model_to_dict(module)
    return info


def save_model_to_json(model: nn.Module, filename: str) -> str:
    model_dict = save_model_to_dict(model)
    if filename is not None:
        with open(filename, "w") as f:
            json.dump(model_dict, f)
    return json.dumps(model_dict)


def load_model(model_path: str) -> nn.Module:
    pass