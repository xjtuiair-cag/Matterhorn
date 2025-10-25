# `matterhorn_pytorch.snn.container`

[回到 `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/7_container.md)

[中文](../../zh_cn/snn/7_container.md)

## 模块简介

SNN 模块的容器，用于组合各个 `matterhorn_pytorch.snn.Module` 。

## `matterhorn_pytorch.snn.Sequential` / `matterhorn_pytorch.snn.container.Sequential`

SNN 序列容器。其与 `nn.Sequential` 用法类似，然而：

（1）其可以接受任何 `torch.nn.Module` 模块，并且对这个模块应用 `matterhorn_pytorch.snn.Module` 独有的方法时，其仅会对其中的 `matterhorn_pytorch.snn.Module` 模块生效。

（2）当模型输出为元组时，默认将元组中的第一个元素作为下一层的输入，同时保存之后的历史状态。

```python
Sequential(
    *args: Tuple[nn.Module]
)
```

### 构造函数参数

`*args (*nn.Module)` ：按空间顺序传入的各个模块。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


model = mth.snn.Sequential(
    mth.snn.Linear(784, 10),
    mth.snn.LIF()
)
print(model)
```