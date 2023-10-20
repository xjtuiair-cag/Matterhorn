from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from typing import List


def get_cpp_files(root_path: str, exceptions: List[str]) -> List[str]:
    full_list = os.listdir(root_path)
    res_list = []
    for filename in full_list:
        if (filename.endswith(".cpp") or filename.endswith(".cu")) and not (filename in exceptions):
            res_list.append(os.path.join(root_path, filename))
    return res_list


setup(
    name="matterhorn_cuda_extensions",
    ext_modules = [
        CUDAExtension(
            "matterhorn_cuda_extensions",
            get_cpp_files(os.path.abspath("."), ["base.cpp"]),
            extra_compile_args = {
                "cxx": ["-g", "-w"],
                "nvcc": ["-O2"]
            }
        ),
    ],
    cmdclass = {
        "build_ext": BuildExtension
    },
)
