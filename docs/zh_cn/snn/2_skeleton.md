# `matterhorn_pytorch.snn.skeleton`

[回到 `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/2_skeleton.md)

[中文](../../zh_cn/snn/2_skeleton.md)

## 模块简介

在该模块中定义了所有 SNN 模块的基类 `matterhorn_pytorch.snn.Module` 。其中 SNN 模块特有的变量与方法均在新定义的类中。 `matterhorn_pytorch.snn.Module` 继承了 `torch.nn.Module` ，并与 `torch.nn.Module` 的用法几乎一致。

## `matterhorn_pytorch.snn.Module` / `matterhorn_pytorch.snn.skeleton.Module`

```python
Module()
```

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


class Demo(mth.snn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return x
```

### 可重载的方法

#### `extra_repr(self) -> str`

用法同 `torch.nn.Module.extra_repr()` ，自定义打印含有模块参数的字符串。

#### `reset(self) -> matterhorn_pytorch.snn.Module`

自定义模型全局重置的时候该如何进行。默认为调用该模块下所有 `matterhorn_pytorch.snn.Module` 子模块的 `reset()` 函数。在执行完所有的时间步之后，您需要将模型进行重置（如将所有神经元的电位重置回静息电位）。此时需要调用该 `reset()` 函数。

#### `detach(self) -> matterhorn_pytorch.snn.Module`

自定义模型计算图的分离。默认为调用该模块下所有 `matterhorn_pytorch.snn.Module` 子模块的 `detach()` 函数。当时间步过长时，保存所有时间步的计算图显然不太合理，此时可以使用 `detach()` 函数将某些时间步的变量从计算图中分离（带有一定的危险性，如果发现 bug 请及时与我们联系）。

#### `step(self, *args: Tuple[torch.Tensor], **kwargs: Mapping[str, Any]) -> matterhorn_pytorch.snn.Module`

部署模块的自定义训练。当该模块不使用反向传播作为训练方式时，需要在外部调用该方法对模型进行训练。默认为调用该模块下所有 `matterhorn_pytorch.snn.Module` 子模块的 `step()` 函数。可以从外部传入参数（如准确率）以自定义模块中权重等变量的更新方式。