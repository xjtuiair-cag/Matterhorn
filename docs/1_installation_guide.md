# Matterhorn的下载与使用

## 1 安装Python、CUDA和PyTorch

### 1.1 安装GIT和Python

**请不要安装3.10及以上版本的Python，因为其不支持1.x版本的PyTorch。若您已安装，可以通过Anaconda创建适合的Python虚拟环境。推荐使用Python3.7~3.9。**

#### Windows用户

首先安装Chocolatey（Windows的包管理器）：

右键右下角Windows图标，选择“命令提示符（管理员）”或“终端（管理员）”（若终端默认为`cmd`），随后输入

```sh
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
```

或选择“PowerShell（管理员）”或“终端（管理员）”（若终端默认为`powershell`），随后输入

```sh
Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
```

按照指示步骤安装即可。安装好后，关闭当前命令行，开启新的命令行。

随后，在命令行中输入

```sh
choco install git
```

当GIT安装完成后，前往[https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.11-Windows-x86_64.exe](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.11-Windows-x86_64.exe)下载并安装Anaconda（基础环境为Python 3.8）。 **注意：** 一定要将conda环境加入PATH（当出现“add ... to PATH”时，请勾选它）。

**测试**

随后，在命令行输入

```sh
git --help
python --help
pip --help
```

分别查看是否安装成功。若无报错，则表明安装成功。

#### MacOS用户

首先安装Homebrew（MacOS的包管理器）：

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

当GIT安装完成后，前往[https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.11-MacOSX-x86_64.pkg](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.11-MacOSX-x86_64.pkg)下载并安装Anaconda（基础环境为Python 3.8）。

**测试**

随后，在命令行输入

```sh
git --help
python --help
pip --help
```

分别查看是否安装成功。若无报错，则表明安装成功。

#### Linux用户

首先更新APT源。在命令行中输入

```sh
sudo apt update
```

更新，随后，在命令行中输入

```sh
sudo apt install git python3-pip python-is-python3
```

安装GIT和PIP。 **注意：** Ubuntu 20.04自带的Python版本为3.8，因此不必重新安装Anaconda；但自带的Python是没有PIP的，需要手动安装。

您也可以前往[https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.11-Linux-x86_64.sh](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.11-Linux-x86_64.sh)下载并通过

```sh
sudo path/to/Anaconda3-2020.11-Linux-x86_64.sh
```

安装Anaconda（基础环境为Python 3.8）。

**测试**

随后，在命令行输入

```sh
git --help
python3 --help
pip3 --help
```

分别查看是否安装成功。若无报错，则表明安装成功。

### 1.2 安装CUDA及CUDNN

若您没有支持CUDA的NVIDIA™显卡，可以跳过此步。 **11.6以上版本的CUDA均可安装，** 推荐安装[CUDA 11.7.1](https://developer.nvidia.com/cuda-11-7-1-download-archive)版本。

#### Linux用户

点击链接[https://developer.nvidia.com/cuda-11-7-1-download-archive](https://developer.nvidia.com/cuda-11-7-1-download-archive)，选择如下选项进行安装：

若您是双系统或虚拟机用户，请进行如下选择：

![双系统或虚拟机请选择如下选项](../assets/0_1.png)

若您是WSL用户，请进行如下选择：

![WSL请选择如下选项](../assets/0_2.png)

随后，页面下方会弹出相关命令行：

![命令行示例](../assets/0_3.png)

请依次复制命令进行CUDA的安装。

在CUDA安装完成之后，请于[https://developer.nvidia.com/zh-cn/cudnn](https://developer.nvidia.com/zh-cn/cudnn)，按照页面指引下载适用于CUDA 11.x的CUDNN并安装。

**请注意：** 虽然执行完以上步骤之后，CUDA及CUDNN均安装完成，但在执行时，它们并不会自动被激活。此时需要我们在`~/.bashrc`中加入自启动代码：

（1）使用`vim`（或文本编辑器等）打开`~/.bashrc`；

```sh
sudo apt install vim
sudo vim ~/.bashrc
```

（2）随后，命令行会弹出`~/.bashrc`中的内容。将光标移至文件末尾，按下`I`键开启编辑，并在结尾处插入如下代码：

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

（3）插入后，按下`esc`键开启命令输入，并输入

```sh
:wq
```

以保存并退出；

（4）关闭命令行或输入

```sh
source ~/.bashrc
```

以使CUDA在命令行中生效。

**测试**

安装好后，在命令行中输入

```sh
nvcc --version
```

若无报错，且观察到版本号为11.7.1，即代表安装成功。

#### Windows用户

请参考教程[https://zhuanlan.zhihu.com/p/99880204](https://zhuanlan.zhihu.com/p/99880204)安装。推荐安装CUDA 11.7.1版本。

**测试**

安装好后，在命令行中输入

```sh
nvcc --version
```

若无报错，且观察到版本号为11.7.1，即代表安装成功。

### 1.3 安装PyTorch

复制自[PyTorch历史版本](https://pytorch.org/get-started/previous-versions/)页面， **请不要安装2.0以上版本的PyTorch，目前我们无法确定是否存在不兼容现象。** 建议安装1.13.1版本。

#### MacOS用户

```sh
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
```

#### Windows或Linux用户

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
# Windows或MacOS
python
# Ubuntu 20.04
python3
```

启动Python，随后输入

```python
import torch
```

，若没有报错，则代表PyTorch已经安装成功。

若您所安装的是CUDA版本的PyTorch，请继续输入

```python
torch.cuda.is_available()
```

若命令行返回

```python
True
```

则代表PyTorch的CUDA扩展是可用的，可以使用GPU进行机器学习。

## 2 安装GCC和G++

Matterhorn依赖C++与CUDA对神经元进行加速。若希望全速运行Matterhorn，获得最佳体验，请安装GCC与G++。

若您在命令行输入

```sh
gcc -v
g++ -v
```

后没有报错，可以跳过此步。

### Windows用户

使用本仓库中提供的`mingw-get-setup.exe`，参考[该教程](https://blog.csdn.net/weixin_43448473/article/details/126942594)安装GCC与G++。

### MacOS用户

参考[该教程](https://segmentfault.com/a/1190000018045211)安装Command Line Tools，其中含有Apple自带的GCC与G++。

### Linux用户

利用APT安装GCC与G++：

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

## 3 安装Matterhorn

克隆仓库：

```sh
cd your/path
git clone https://github.com/AmperiaWang/Matterhorn.git
```

随后安装：

```sh
cd Matterhorn
python setup.py install
```

如果发生报错，请检查是否在管理员模式下运行（Windows打开终端的管理员模式后重新执行，Linux在命令前加入`sudo`）。

**测试**

打开命令行，输入

```sh
python
```

打开Python，随后输入

```python
import matterhorn_pytorch
import matterhorn_cpp_extensions # 若没安装G++会报错
import matterhorn_cuda_extensions # 若没安装CUDA会报错，若没有Nvidia显卡可以忽略
```

如果没有报错，则表明Matterhorn已经安装成功。