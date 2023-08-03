from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


requirements = ["torch"]


with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()


with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    install_requires = install_requires,
    name = "matterhorn",
    version = "0.1.0",
    author = "CAG, IAIR, XJTU, Xi'an, China",
    author_email = "ericwang017@stu.xjtu.edu.cn",
    description = "A neuromorphic framework based on PyTorch.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "",
    packages = find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3 :: Only",
        "License :: MIT No Attribution License",
        "Operating System :: OS Independent",
    ],
    ext_modules = [
        CUDAExtension(
            "matterhorn_cuda",
            [
                "matterhorn/cuda/api.cpp",
                "matterhorn/cuda/stdp.cpp",
                "matterhorn/cuda/stdp_cuda.cu"
            ],
            extra_compile_args = {
                "cxx": ["-g"],
                "nvcc": ["-O2"]
            }
        ),
    ],
    cmdclass = {
        "build_ext": BuildExtension
    },
    python_requires = '>=3.7'
)