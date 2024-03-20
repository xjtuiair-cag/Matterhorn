# -*- coding: UTF-8 -*-
"""
液体状态机的相关函数。
包括邻接矩阵的生成，液体状态机的融合与分离等。
"""


import torch
import torch.nn as nn
from matterhorn_pytorch.lsm.layer import LSM
from typing import Iterable
try:
    from rich import print
except:
    pass


def init_adjacent_direct(neuron_num: int, axon_numbers: Iterable[int], dendrite_numbers: Iterable[int]) -> torch.Tensor:
    """
    自定义连接关系生成邻接矩阵。
    Args:
        neuron_num (int): 液体状态机中神经元的个数
        axon_numbers (int*): 各个轴突（输出）的编号
        dendrite_numbers (int*): 各个树突（输入）的编号
    Returns:
        res (torch.Tensor): 邻接矩阵
    """
    res = torch.zeros(neuron_num, neuron_num, dtype = torch.float)
    res[axon_numbers, dendrite_numbers] = 1.0
    return res


def init_adjacent_uniform(neuron_num: int, threshold: float = 0.75) -> torch.Tensor:
    """
    由均匀分布生成随机邻接矩阵。
    Args:
        neuron_num (int): 液体状态机中神经元的个数
        threshold (float): 连接的阈值，阈值越大代表连接越容易
    Returns:
        res (torch.Tensor): 邻接矩阵
    """
    res = torch.rand(neuron_num, neuron_num)
    return (res <= threshold).to(torch.float)


def init_adjacent_norm(neuron_num: int, threshold: float = 0.75) -> torch.Tensor:
    """
    由正态分布生成随机邻接矩阵。
    Args:
        neuron_num (int): 液体状态机中神经元的个数
        threshold (float): 连接的阈值，阈值越大代表连接越容易
    Returns:
        res (torch.Tensor): 邻接矩阵
    """
    res = torch.randn(neuron_num, neuron_num)
    return (res <= threshold).to(torch.float)


def init_adjacent_dist_1d(length: int, threshold: float = 1.0) -> torch.Tensor:
    """
    由神经元之间的相对（一维）距离得到神经元的邻接矩阵。相对距离越近神经元的连接概率越高。
    Args:
        length (int): 液体状态机中神经元的个数
        threshold (float): 连接的阈值，阈值越大代表连接越容易
    Returns:
        res (torch.Tensor): 邻接矩阵
    """
    neuron_num = length
    res = torch.zeros(neuron_num, neuron_num, dtype = torch.float)
    position = torch.arange(neuron_num).to(torch.float)
    res = torch.abs((res.T + position).T - position)
    res = (res + 1) * torch.rand_like(res)
    return (res <= threshold).to(torch.float)


def init_adjacent_dist_2d(height: int, width: int, threshold: float = 1.0) -> torch.Tensor:
    """
    由神经元之间的相对（二维）距离得到神经元的邻接矩阵。相对距离越近神经元的连接概率越高。
    Args:
        height (int): 二维高度，矩阵中的行数
        width (int): 二维宽度，矩阵中的列数
        threshold (float): 连接的阈值，阈值越大代表连接越容易
    Returns:
        res (torch.Tensor): 邻接矩阵
    """
    neuron_num = height * width
    res = torch.zeros(neuron_num, neuron_num, dtype = torch.float)
    position = torch.arange(neuron_num).to(torch.float)
    position_y = position // width
    position_x = position % width
    res_y = torch.abs((res.T + position_y).T - position_y)
    res_x = torch.abs((res.T + position_x).T - position_x)
    res = torch.sqrt(res_y ** 2 + res_x ** 2)
    res = (res + 1) * torch.rand_like(res)
    return (res <= threshold).to(torch.float)


def init_adjacent_dist_3d(channels: int, height: int, width: int, threshold: float = 1.0) -> torch.Tensor:
    """
    由神经元之间的相对（三维）距离得到神经元的邻接矩阵。相对距离越近神经元的连接概率越高。
    Args:
        channels (int): 三维通道数
        height (int): 三维高度，矩阵中的行数
        width (int): 三维宽度，矩阵中的列数
        threshold (float): 连接的阈值，阈值越大代表连接越容易
    Returns:
        res (torch.Tensor): 邻接矩阵
    """
    neuron_num = channels * height * width
    res = torch.zeros(neuron_num, neuron_num, dtype = torch.float)
    position = torch.arange(neuron_num).to(torch.float)
    position_p = position // (width * height)
    position_yx = position - position_p * width * height
    position_y = position_yx // width
    position_x = position_yx % width
    res_p = torch.abs((res.T + position_p).T - position_p)
    res_y = torch.abs((res.T + position_y).T - position_y)
    res_x = torch.abs((res.T + position_x).T - position_x)
    res = torch.sqrt(res_p ** 2 + res_y ** 2 + res_x ** 2)
    res = (res + 1) * torch.rand_like(res)
    return (res <= threshold).to(torch.float)


def merge(*models: LSM, connection: str = "uniform", threshold: float = 0.75) -> LSM:
    new_neuron_num = sum([m.neuron_num for m in models])
    adjacent = torch.zeros(new_neuron_num, new_neuron_num)
    if connection == "norm":
        adjacent = init_adjacent_norm(new_neuron_num, threshold)
    elif connection == "uniform":
        adjacent = init_adjacent_uniform(new_neuron_num, threshold)
    offset = 0
    soma = None
    multi_time_step = None
    reset_after_process = None
    for m in models:
        adjacent[offset:offset + m.neuron_num, offset:offset + m.neuron_num] = m.adjacent
        offset += m.neuron_num
        soma = m.soma # TODO: different neurons in the same LSM
        assert multi_time_step is None or multi_time_step == m.multi_time_step, "Not all models have same step mode."
        multi_time_step = m.multi_time_step
        assert reset_after_process is None or reset_after_process == m.reset_after_process, "Not all models have same reset mode."
        reset_after_process = m.reset_after_process
    res = LSM(adjacent, soma, multi_time_step, reset_after_process, m.adjacent.device)
    return res