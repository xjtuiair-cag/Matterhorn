# 开发者须知

## 1 安装

![logo](../assets/logo.png)

若有意开发Matterhorn，请按照如下方式配置您的环境：

### 1.1 安装Ubuntu 20.04

[双系统](https://zhuanlan.zhihu.com/p/363640824)、[WSL](https://blog.csdn.net/weixin_43577131/article/details/111928160)或[虚拟机](https://blog.csdn.net/qq_45373920/article/details/122409002)均可，WSL要安装[MobaXTerm](https://mobaxterm.mobatek.net/)并输入如下命令配置GUI：

```sh
sudo apt install x11-apps
```

### 1.2 安装CUDA及CUDNN

若您没有支持CUDA的NVIDIA™显卡，可以跳过此步。推荐安装[CUDA 11.7](https://developer.nvidia.com/cuda-11-7-1-download-archive)版本。

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

### 1.3 安装PyTorch

复制自[PyTorch历史版本](https://pytorch.org/get-started/previous-versions/)页面，建议安装1.13.1版本作为开发。

如果您是MacOSX用户，请输入以下命令安装：

```sh
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
```

如果您是Windows或Linux用户，请在以下命令中选择合适的一个安装：

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

当安装好后，打开命令行，输入

```sh
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

### 1.4 在本地部署Matterhorn的GIT仓库

不同于`README`中所述`git clone`的方式，若希望进一步开发，我们需要在本地初始化GIT仓库，随后与远端同步。输入

```sh
git init
```

初始化新的GIT仓库后，输入

```sh
git remote add origin https://github.com/AmperiaWang/Matterhorn.git
```

将本地仓库与远程仓库相链接。

执行

```sh
git pull origin main
```

可以将最新的代码由远端的`main`分支拉取至本地仓库。

**注意：** 在第一次拉取时，请执行

```sh
git branch -M master main
```

以将本地的分支由`master`修改为`main`，与远端分支同步。

当代码修改完成后，请执行

```sh
git add .
git commit -m "[修改说明]"
```

以将所执行的修改提交到本地分支之中。若要与远端分支同步修改，请先执行

```sh
git pull origin main
```

以与远端分支中其他人提交的代码合并并解决您与其他人在代码上的冲突，随后执行

```sh
git push origin main
```

将您的代码提交至远端仓库。