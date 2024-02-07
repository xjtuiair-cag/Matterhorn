# -*- coding: UTF-8 -*-
"""
（未完工，测试功能）SNN模型的保存与提取。
"""


import io
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.onnx


def save(filename: str, model: nn.Module, dummy_input: torch.Tensor) -> None:
    torch.onnx.export(
        model = model,
        args = (dummy_input,),
        f = filename,
    )