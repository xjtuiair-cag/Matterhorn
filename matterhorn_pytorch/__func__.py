import os
from subprocess import run
import torch


def cpp_available() -> bool:
    """
    判断C++扩展是否可用。
    Returns:
        is_available (bool): C++扩展是否可用
    """
    try:
        import matterhorn_cpp_extensions as __mth_cppext__
        del __mth_cppext__
        return True
    except:
        return False


def cuda_available() -> bool:
    """
    判断CUDA扩展是否可用。
    Returns:
        is_available (bool): CUDA扩展是否可用
    """
    if not torch.cuda.is_available():
        return False
    try:
        import matterhorn_cuda_extensions as __mth_cuext__
        del __mth_cuext__
        return True
    except:
        return False


def transpose(x: torch.Tensor) -> torch.Tensor:
    """
    转置一个张量。
    Args:
        x (torch.Tensor): 转置前的张量
    Returns:
        x (torch.Tensor): 转置后的张量
    """
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))