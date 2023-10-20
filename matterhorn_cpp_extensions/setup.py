from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
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
    name="matterhorn_cpp_extensions",
    ext_modules = [
        CppExtension(
            "matterhorn_cpp_extensions",
            get_cpp_files(".", ["base.cpp", "base.cu"]),
            extra_compile_args = {
                "cxx": ["-g", "-w"]
            }
        ),
    ],
    cmdclass = {
        "build_ext": BuildExtension
    },
)