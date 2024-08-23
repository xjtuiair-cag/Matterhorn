from setuptools import find_packages
from setuptools import setup
import os

requirements = ["torch"]


with open(os.path.join(os.path.abspath("."), "requirements.txt"), "r", encoding="utf-8") as fh:
    install_requires = fh.read()


with open(os.path.join(os.path.abspath("."), "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    install_requires = install_requires,
    name = "matterhorn_pytorch",
    version = "1.4.1",
    author = "CAG, IAIR, XJTU, Xi'an, China",
    author_email = "ericwang017@stu.xjtu.edu.cn",
    description = "Matterhorn is a novel general neuromorphic computing framework based on PyTorch.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license = "License :: OSI Approved :: MIT License",
    url = "https://github.com/xjtuiair-cag/Matterhorn",
    packages = find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.7',
    # ext_modules = ext_modules,
    # cmdclass = cmdclass
)