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
    name = "matterhorn",
    version = "1.0.0",
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
    python_requires = '>=3.7'
)