# `matterhorn_pytorch.util.converter`

[回到 `matterhorn_pytorch.util`](./README.md)

[English](../../en_us/util/2_converter.md)

[中文](../../zh_cn/util/2_converter.md)

## 模块简介

ANN 转 SNN 的基本原理为：在 SNN 模型中，以软重置为重置机制的 IF 神经元的输出脉冲发射率与输入电位间近似成正比关系。因此，可以使用以软重置为重置机制的 IF 神经元作为 SNN 模型中的“激活函数”。其具体步骤为：

（1）将数据集（一般为训练集）输入 ANN 中，统计每层 ReLU 函数输出的最大值 $\lambda^{l}$ ；

（2）将 ANN 模块替换成 Matterhorn 对应的 SNN 模块（若是未知模块，则用 `matterhorn_pytorch.snn.container.Proxy` 包裹），随后使用 `matterhorn_pytorch.snn.container.Temporal` 将其转为多时间步的 SNN 模块；

（3）以计算图顺序遍历 SNN 模型：

[1] 若有“全连接/卷积→批归一化”的对应结构，应用公式

$$\tilde{W}_{ij}^{l}=\frac{\gamma_{i}^{l}}{\sigma_{i}^{l}}W_{ij}^{l}$$

$$\tilde{b}_{i}^{l}=\frac{\gamma_{i}^{l}}{\sigma_{i}^{l}}(b_{i}^{l}-\mu_{i}^{l})+\beta_{i}^{l}$$

更新权重 $W_{ij}^{l}$ 和偏置 $b_{i}^{l}$。其中 $\mu_{i}$ 和 $\sigma_{i}^{l}$ 分别为输入的均值和标准差；$\gamma_{i}^{l}$ 和 $\beta_{i}^{l}$ 为当前层的仿射参数。更新完权重后，清除批归一化模块；

[2] 若有“全连接/卷积→激活函数”的对应结构，应用公式

$$W_{ij}^{l} \rightarrow \frac{W_{ij}^{l}}{\lambda^{l}}$$

$$b_{i}^{l} \rightarrow \frac{b_{i}^{l}}{\lambda^{l}}$$

更新权重 $W_{ij}^{l}$ 和偏置 $b_{i}^{l}$；

[3] 若有“激活函数→全连接/卷积”的对应结构，应用公式

$$W_{ij}^{l} \rightarrow \lambda^{l-1}W_{ij}^{l}$$

更新权重 $W_{ij}^{l}$。

## `matterhorn_pytorch.util.converter.ann_to_snn`

ANN 转 SNN 的转换器。

```python
ann_to_snn(
    model: nn.Module,
    demo_data: torch.utils.data.Dataset,
    mode: str = "max"
) -> snn.Module
```

### 参数

`model (torch.nn.Module)`：待转的 ANN 模型。

`demo_data (torch.utils.data.Dataset)`：示例数据，用于确定每个激活函数输出的最大值。

`mode (str)`：如何确定转换倍率 $\lambda^{l}$。默认为 `"max"`，即每个激活函数的最大值。

### 返回值

`snn_model (matterhorn_pytorch.snn.Module)`：转换后的 SNN 模型。

### 示例用法

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets.mnist import MNIST
from matterhorn_pytorch.util.converter import ann_to_snn

class ResidualConnection(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.res = ResidualConnection()
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv3(x)
        z = self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x)))))
        y = self.res(w, z)
        y = self.relu2(y)
        return y

if __name__ == "__main__":
    ann_model = nn.Sequential(
        ResBlock(1, 8),
        ResBlock(8, 16),
    )
    data = MNIST(
        root = "./examples/data/",
        train = False,
        transform = torchvision.transforms.ToTensor(),
        download = True
    )
    torch.save(a, "ann_model.pt")
    snn_model = ann_to_snn(ann_model, data)
    torch.save(s, "snn_model.pt")
```

**注意：** 待转 ANN 模型的所有模块，其 `forward` 函数内要么全为算子的调用，要么全为模块的调用。 **最好不要将算子与模块合并调用（尤其是含有全连接/卷积/激活函数的模块），否则可能无法顺利将 ANN 模型转为 SNN 模型！**

$\times$ 错误示例：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    w = self.conv3(x)
    z = self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x)))))
    y = w + z # 可能会导致追踪计算图时出现异常
    y = self.relu2(y)
    return y
```

$\checkmark$ 正确示例：

```python
def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
```

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    w = self.conv3(x)
    z = self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x)))))
    y = self.res(w, z)
    y = self.relu2(y)
    return y
```