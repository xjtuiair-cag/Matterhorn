# Matterhorn 的下载与使用

[English](../en_us/1_install.md)

[中文](../zh_cn/1_install.md)

## 1 安装 Python 、 CUDA 和 PyTorch

### 1.1 安装 GIT 和 Python

**请不要安装 3.10 及以上版本的 Python ，因为其不支持 1.x 版本的 PyTorch 。若您已安装，可以通过 Anaconda 创建适合的 Python 虚拟环境。推荐使用 Python 3.7~3.9 。**

#### Windows 用户

首先安装 Chocolatey （Windows 的包管理器）：

右键右下角 Windows 图标，选择“命令提示符（管理员）”或“终端（管理员）”（若终端默认为 `cmd`），随后输入

```sh
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
```

或选择“PowerShell （管理员）”或“终端（管理员）”（若终端默认为 `powershell`），随后输入

```sh
Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
```

按照指示步骤安装即可。安装好后，关闭当前命令行，开启新的命令行。

随后，在命令行中输入

```sh
choco install git
```

当 GIT 安装完成后，前往 [https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.11-Windows-x86_64.exe](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.11-Windows-x86_64.exe) 下载并安装 Anaconda （基础环境为 Python 3.8）。 **注意：** 一定要将 conda 环境加入 PATH （当出现“add ... to PATH”时，请勾选它）。

**测试**

随后，在命令行输入

```sh
git --help
python --help
pip --help
```

分别查看是否安装成功。若无报错，则表明安装成功。

#### MacOS 用户

首先安装 Homebrew （MacOS 的包管理器）：

打开“终端”应用，输入

```sh
# 国外（或科学上网）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# 国内
/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
```

按照指示步骤安装即可。安装好后，关闭当前命令行，开启新的命令行。

随后，在命令行中输入

```sh
brew install git
```

当 GIT 安装完成后，前往 [https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.11-MacOSX-x86_64.pkg](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.11-MacOSX-x86_64.pkg) 下载并安装 Anaconda （基础环境为 Python 3.8）。

**测试**

随后，在命令行输入

```sh
git --help
python --help
pip --help
```

分别查看是否安装成功。若无报错，则表明安装成功。

#### Linux 用户

首先更新 APT 源。在命令行中输入

```sh
sudo apt update
```

更新，随后，在命令行中输入

```sh
sudo apt install git python3-pip python-is-python3
```

安装 GIT 和 PIP 。 **注意：** Ubuntu 20.04 自带的 Python 版本为 3.8 ，因此不必重新安装 Anaconda ；但自带的 Python 是没有 PIP 的，需要手动安装。

您也可以前往 [https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.11-Linux-x86_64.sh](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.11-Linux-x86_64.sh) 下载并通过

```sh
sudo path/to/Anaconda3-2020.11-Linux-x86_64.sh
```

安装 Anaconda （基础环境为 Python 3.8）。

**测试**

随后，在命令行输入

```sh
git --help
python3 --help
pip3 --help
```

分别查看是否安装成功。若无报错，则表明安装成功。

### 1.2 安装 CUDA 及 CUDNN

若您没有支持 CUDA 的 NVIDIA™ 显卡，可以跳过此步。 **11.6 以上版本的 CUDA 均可安装，** 推荐安装 [CUDA 11.7.1](https://developer.nvidia.com/cuda-11-7-1-download-archive) 版本。

#### Linux 用户

点击链接 [https://developer.nvidia.com/cuda-11-7-1-download-archive](https://developer.nvidia.com/cuda-11-7-1-download-archive) ，选择如下选项进行安装：

若您是双系统或虚拟机用户，请进行如下选择：

![双系统或虚拟机请选择如下选项](../../assets/docs/install_1.png)

若您是 WSL 用户，请进行如下选择：

![WSL请选择如下选项](../../assets/docs/install_2.png)

随后，页面下方会弹出相关命令行：

![命令行示例](../../assets/docs/install_3.png)

请依次复制命令进行 CUDA 的安装。

在 CUDA 安装完成之后，请于 [https://developer.nvidia.com/zh-cn/cudnn](https://developer.nvidia.com/zh-cn/cudnn) ，按照页面指引下载适用于 CUDA 11.x 的 CUDNN 并安装。

**请注意：** 虽然执行完以上步骤之后， CUDA 及 CUDNN 均安装完成，但在执行时，它们并不会自动被激活。此时需要我们在 `~/.bashrc` 中加入自启动代码：

（1）使用 `vim` （或文本编辑器等）打开 `~/.bashrc` ；

```sh
sudo apt install vim
sudo vim ~/.bashrc
```

（2）随后，命令行会弹出 `~/.bashrc` 中的内容。将光标移至文件末尾，按下 `I` 键开启编辑，并在结尾处插入如下代码：

```sh
function switch_cuda {
   v=$1
   export PATH=/usr/local/cuda-$v/bin:$PATH
   export CUDADIR=/usr/local/cuda-$v
   export CUDA_HOME=/usr/local/cuda-$v
   export LD_LIBRARY_PATH=/usr/local/cuda-$v/lib64:$LD_LIBRARY_PATH
}
switch_cuda 11.7
```

（3）插入后，按下 `esc` 键开启命令输入，并输入

```sh
:wq
```

以保存并退出；

（4）关闭命令行或输入

```sh
source ~/.bashrc
```

以使 CUDA 在命令行中生效。

**测试**

安装好后，在命令行中输入

```sh
nvcc --version
```

若无报错，且观察到版本号为 11.7.1 ，即代表安装成功。

#### Windows 用户

请参考教程 [https://zhuanlan.zhihu.com/p/99880204](https://zhuanlan.zhihu.com/p/99880204) 安装。推荐安装 CUDA 11.7.1 版本。

**测试**

安装好后，在命令行中输入

```sh
nvcc --version
```

若无报错，且观察到版本号为 11.7.1 ，即代表安装成功。

### 1.3 安装 PyTorch

复制自 [PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)页面， **请不要安装 2.0 以上版本的 PyTorch ，目前我们无法确定是否存在不兼容现象。** 建议安装 1.13.1 版本。

#### MacOS 用户

```sh
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
```

#### Windows 或 Linux 用户

```sh
# 对 ROCM 5.2 （AMD™显卡的机器学习库，仅限Linux或WSL）
pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
# 对 CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# 对 CUDA 11.7 （若已安装CUDA，选择这个版本）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# 对 CPU（若无法安装CUDA，选择这个版本）
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

**测试**

当安装好后，打开命令行，输入

```sh
# Windows 或 MacOS
python
# Ubuntu 20.04
python3
```

启动 Python ，随后输入

```python
import torch
```

，若没有报错，则代表 PyTorch 已经安装成功。

若您所安装的是 CUDA 版本的 PyTorch ，请继续输入

```python
torch.cuda.is_available()
```

若命令行返回

```python
True
```

则代表 PyTorch 的 CUDA 扩展是可用的，可以使用 GPU 进行机器学习。

## 2 安装 GCC 和 G++

Matterhorn 依赖 C++ 与 CUDA 对神经元进行加速。若希望全速运行 Matterhorn ，获得最佳体验，请安装 GCC 与 G++ 。

若您在命令行输入

```sh
gcc -v
g++ -v
```

后没有报错，可以跳过此步。

### Windows 用户

使用本仓库中提供的 `mingw-get-setup.exe` ，参考[该教程](https://blog.csdn.net/weixin_43448473/article/details/126942594)安装 GCC 与 G++ 。

### MacOS 用户

参考[该教程](https://segmentfault.com/a/1190000018045211)安装 Command Line Tools ，其中含有 Apple 自带的 GCC 与 G++ 。

### Linux 用户

利用 APT 安装 GCC 与 G++ ：

```sh
sudo apt install gcc g++
```

**测试**

打开命令行，输入

```sh
gcc -v
g++ -v
```

若没有报错，则代表成功安装。

## 3 安装 Matterhorn

克隆仓库：

```sh
cd your/path
git clone https://github.com/xjtuiair-cag/Matterhorn.git
```

随后安装：

```sh
cd Matterhorn
python setup.py develop
```

如果发生报错，请检查是否在管理员模式下运行（Windows 打开终端的管理员模式后重新执行， Linux 在命令前加入 `sudo`）。

**测试**

打开命令行，输入

```sh
python
```

打开 Python ，随后输入

```python
import matterhorn_pytorch as mth
```

如果没有报错，则表明 Matterhorn 已经安装成功。