import os
from subprocess import run
import torch


def cpp_available() -> bool:
    try:
        import matterhorn_cpp_extensions as __mth_cppext__
        del __mth_cppext__
        return True
    except:
        return False


def cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import matterhorn_cuda_extensions as __mth_cuext__
        del __mth_cuext__
        return True
    except:
        return False