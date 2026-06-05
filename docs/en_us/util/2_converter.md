# `matterhorn_pytorch.util.converter`

[Back to `matterhorn_pytorch.util`](./README.md)

[English](../../en_us/util/2_converter.md)

[中文](../../zh_cn/util/2_converter.md)

## Module Introduction

The basic principle of ANN-to-SNN conversion is that in an SNN model, the output spike firing rate of an IF neuron with soft reset is approximately proportional to its input membrane potential. Therefore, IF neurons with soft reset can serve as the "activation function" in the SNN model. The conversion steps are:

(1) Feed the dataset (usually the training set) into the ANN and record the maximum ReLU output $\lambda^{l}$ at each layer;

(2) Replace ANN modules one by one with the corresponding Matterhorn SNN modules. For custom modules that cannot be mapped automatically, provide a replacement rule via the `replace_rule` parameter of `ann_to_snn`. Since Matterhorn 2.0.0, all SNN modules run in multi-step mode by default and require no extra wrapping;

(3) Traverse the SNN model in computation-graph order:

[1] For a "linear/convolution → batch normalization" structure, apply

$$\tilde{W}_{ij}^{l}=\frac{\gamma_{i}^{l}}{\sigma_{i}^{l}}W_{ij}^{l}$$

$$\tilde{b}_{i}^{l}=\frac{\gamma_{i}^{l}}{\sigma_{i}^{l}}(b_{i}^{l}-\mu_{i}^{l})+\beta_{i}^{l}$$

to update weights $W_{ij}^{l}$ and biases $b_{i}^{l}$, where $\mu_{i}$ and $\sigma_{i}^{l}$ are the input mean and standard deviation, and $\gamma_{i}^{l}$ and $\beta_{i}^{l}$ are the affine parameters of the current layer. After updating, remove the batch normalization module;

[2] For a "linear/convolution → activation" structure, apply

$$W_{ij}^{l} \rightarrow \frac{W_{ij}^{l}}{\lambda^{l}}$$

$$b_{i}^{l} \rightarrow \frac{b_{i}^{l}}{\lambda^{l}}$$

to update weights $W_{ij}^{l}$ and biases $b_{i}^{l}$;

[3] For an "activation → linear/convolution" structure, apply

$$W_{ij}^{l} \rightarrow \lambda^{l-1}W_{ij}^{l}$$

to update weights $W_{ij}^{l}$.

## `matterhorn_pytorch.util.converter.ann_to_snn`

ANN-to-SNN converter.

```python
ann_to_snn(
    model: nn.Module,
    demo_data: torch.utils.data.Dataset,
    pre_process: Callable = lambda x: x[0],
    replace_rule: Optional[Callable] = None,
    mode: str = "max"
) -> snn.Module
```

### Parameters

`model (torch.nn.Module)`: The ANN model to convert.

`demo_data (torch.utils.data.Dataset)`: Sample data used to determine the maximum output of each activation function.

`pre_process (Callable)`: Preprocessing function that extracts a single sample from the dataset and converts it into a model input tensor. Defaults to `lambda x: x[0]`.

`replace_rule (Callable | None)`: Custom module replacement rule. If it returns a non-`None` SNN module, that module is used instead of the default mapping.

`mode (str)`: How to determine the scaling factor $\lambda^{l}$. Defaults to `"max"`, i.e. the maximum value of each activation function.

### Returns

`snn_model (matterhorn_pytorch.snn.Module)`: The converted SNN model.

### Example Usage

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
    torch.save(ann_model, "ann_model.pt")
    snn_model = ann_to_snn(ann_model, data)
    torch.save(snn_model, "snn_model.pt")
```

**Note:** In every module of the ANN to be converted, the `forward` function should call either only operators or only submodules. **Avoid mixing operator calls with submodule calls (especially for modules containing linear/convolution/activation layers), otherwise the ANN may fail to convert to an SNN!**

$\times$ Incorrect example:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    w = self.conv3(x)
    z = self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x)))))
    y = w + z # May cause exceptions when tracing the computation graph
    y = self.relu2(y)
    return y
```

$\checkmark$ Correct example:

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
