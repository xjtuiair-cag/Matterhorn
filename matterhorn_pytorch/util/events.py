import torch
from typing import Iterable


def event_seq_to_tensor(event_seq: torch.Tensor, shape: Iterable = None, original_shape: Iterable = None, count: bool = False) -> torch.Tensor:
    """
    将事件序列转为事件张量。
    Args:
        event_seq (torch.Tensor): 事件序列，形状为[N, A]
        shape (int*): 输出事件张量的形状
        original_shape (int*): 事件张量原本的形状，若置空，则视为与事件张量大小一致
        count (bool): 是否输出事件计数，默认为False，即只输出脉冲（0或1）
    Return:
        event_tensor (torch.Tensor): 事件张量，形状为[T, C, ...]
    """
    if not isinstance(event_seq, torch.Tensor):
        event_seq = torch.tensor(event_seq, dtype = torch.int)
    event_seq = event_seq.to(torch.float)
    if shape is None:
        shape = tuple([torch.max(event_seq[:, idx]) for idx in event_seq.shape[1]])
    if original_shape is None:
        original_shape = shape
    event_tensor = torch.zeros(*shape, dtype = torch.float)
    if not event_seq.shape[0]:
        return event_tensor
    for idx in range(len(shape)):
        event_seq[:, idx] = torch.floor(event_seq[:, idx] * shape[idx] / original_shape[idx])
        event_filter = (event_seq[:, idx] >= 0) & (event_seq[:, idx] < shape[idx])
        event_seq = event_seq[event_filter]
    event_seq, counts = torch.unique(event_seq, dim = 0, return_counts = True)
    event_seq = event_seq.to(torch.long)
    counts = counts.to(torch.float)
    event_tensor[event_seq.permute(1, 0).tolist()] = (counts if count else 1.0)
    return event_tensor


def event_tensor_to_seq(event_tensor: torch.Tensor) -> torch.Tensor:
    """
    将事件张量转为事件序列。
    Args:
        event_tensor (torch.Tensor): 事件张量，形状为[T, C, ...]
    Return:
        event_seq (torch.Tensor): 事件序列，形状为[N, A]
    """
    if not isinstance(event_tensor, torch.Tensor):
        event_tensor = torch.tensor(event_tensor, dtype = torch.float)
    event_seq = event_tensor.nonzero()
    return event_seq