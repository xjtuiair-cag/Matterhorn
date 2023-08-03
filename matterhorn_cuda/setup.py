from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="matterhorn_cuda",
    ext_modules = [
        CUDAExtension(
            "matterhorn_cuda",
            [
                "api.cpp",
                "stdp.cpp",
                "stdp_cuda.cu"
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
)
