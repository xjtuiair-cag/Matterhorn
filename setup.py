from setuptools import find_packages
from setuptools import setup


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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6'
)