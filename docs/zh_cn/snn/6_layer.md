# `matterhorn_pytorch.snn.layer`

[回到 `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/6_layer.md)

[中文](../../zh_cn/snn/6_layer.md)

## 模块简介

本模块定义了一些完整的神经网络层或层间操作，如池化、展开等。

## `matterhorn_pytorch.snn.layer.Layer`

```python
Layer(
    batch_first: bool = False
)
```

### 构造函数参数

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 可重载的方法

#### `forward(self, o: torch.Tensor) -> torch.Tensor`

与 `nn.Module.forward` 一致。

## `matterhorn_pytorch.snn.STDPLinear` / `matterhorn_pytorch.snn.layer.STDPLinear`

采用脉冲时序依赖可塑性（STDP）学习机制的全连接层。 STDP 作为常见的基于脉冲的学习机制，其权重更新遵循公式：

$$\Delta w_{ij}=\sum_{t_{j}}{\sum_{t_{i}}W(t_{i}-t_{j})}$$

其中权重函数 $W(x)$ 为：

$$
W(x)=
\left \{
\begin{aligned}
A_{+}e^{-\frac{x}{\tau_{+}}},x>0 \\\\
0,x=0 \\\\
-A_{-}e^{\frac{x}{\tau_{-}}},x<0
\end{aligned}
\right .
$$

， $t_{i}$ 为索引为 $i$ 的神经元输入脉冲产生的时间步， $t_{j}$ 为索引为 $j$ 的突触输入脉冲到达的时间步。

```python
STDPLinear(
    soma: torch.nn.Module,
    in_features: int,
    out_features: int,
    a_pos: float = 0.015,
    tau_pos: float = 2.0,
    a_neg: float = 0.015,
    tau_neg: float = 2.0,
    device: torch.device = None,
    dtype: torch.dtype = None
)
```

### 构造函数参数

`soma (torch.nn.Module)` ：采用哪种胞体。可选择的胞体类型可以参考模块 [`matterhorn_pytorch.snn.soma`](./4_soma.md) 。

`in_features (int)` ：输入的长度 `I` 。输入的形状为 `[B, I]` （单时间步模式） 或 `[T, B, I]` （多时间步模式）。

`out_features (int)` ：输出的长度 `O` 。输出的形状为 `[B, O]` （单时间步模式） 或 `[T, B, O]` （多时间步模式）。

`a_pos (float)` ： STDP 参数 $A_{+}$ 。

`tau_pos (float)` ： STDP 参数 $\tau_{+}$ 。

`a_neg (float)` ： STDP 参数 $A_{-}$ 。

`tau_neg (float)` ： STDP 参数 $\tau_{-}$ 。

`device (torch.device)` ：计算所使用的计算设备。

`dtype (torch.dtype)` ：计算所使用的数据类型。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


l1 = mth.snn.STDPLinear(mth.snn.LIF(), 784, 10) # [T, B, 784] -> [T, B, 10]
```

## `matterhorn_pytorch.snn.STDPConv2d` / `matterhorn_pytorch.snn.layer.STDPConv2d`

采用脉冲时序依赖可塑性（STDP）学习机制的2维卷积层。

```python
STDPConv2d(
    soma: torch.nn.Module,
    in_channels: int,
    out_channels: int,
    kernel_size: _size_2_t,
    stride: _size_2_t = 1,
    padding: _size_2_t = 0,
    dilation: _size_2_t = 1,
    a_pos: float = 0.0002,
    tau_pos: float = 2.0,
    a_neg: float = 0.0002,
    tau_neg: float = 2.0,
    device: torch.device = None,
    dtype: torch.dtype = None
)
```

### 构造函数参数

`soma (torch.nn.Module)` ：采用哪种胞体。可选择的胞体类型可以参考模块 [`matterhorn_pytorch.snn.soma`](./4_soma.md) 。

`in_channels (int)` ：输入通道数 `CI` 。输入的形状为 `[B, CI, HI, WI]` （单时间步模式） 或 `[T, B, CI, HI, WI]` （多时间步模式）。

`out_channels (int)` ：输出通道数 `CO` 。输入的形状为 `[B, CO, HO, WO]` （单时间步模式） 或 `[T, B, CO, HO, WO]` （多时间步模式）。

`kernel_size (size_2_t)` ：卷积核的形状。

`stride (size_2_t)` ：步长。在原图经过多少个像素后进行卷积。

`padding (size_2_t | str)` ：边界大小。在边缘填充多少空白。

`dilation (size_2_t)` ：在卷积时，每隔多少像素进行一次乘加操作。

`a_pos (float)` ： STDP 参数 $A_{+}$ 。

`tau_pos (float)` ： STDP 参数 $\tau_{+}$ 。

`a_neg (float)` ： STDP 参数 $A_{-}$ 。

`tau_neg (float)` ： STDP 参数 $\tau_{-}$ 。

`device (torch.device)` ：计算所使用的计算设备。

`dtype (torch.dtype)` ：计算所使用的数据类型。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


l1 = mth.snn.STDPConv2d(mth.snn.LIF(), 2, 32, 3, 2, 1) # [T, B, 2, 32, 32] -> [T, B, 32, 16, 16]
```

## `matterhorn_pytorch.snn.MaxPool1d` / `matterhorn_pytorch.snn.layer.MaxPool1d`

一维最大池化层。将对脉冲的最大池化定义为：只要有任意一个输入产生脉冲，则输出产生脉冲。可以用如下公式描述：

$$O_{i}^{l}(t)=(\sum_{j=1}^{m}{O_{j}^{l-1}(t)} \ge 1)$$

其中 $i$ 为输出索引， $j$ 为输出索引为 $i$ 时应被统计的输入索引， $m$ 为应被统计的输入索引个数。

```python
MaxPool1d(
    kernel_size: _size_any_t,
    stride: Optional[_size_any_t] = None,
    padding: _size_any_t = 0,
    dilation: _size_any_t = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
    batch_first: bool = False
)
```

### 构造函数参数

`kernel_size (size_any_t)` ：池化核的大小。

`stride (size_any_t | None)` ：一次池化操作跨越多少像素。

`padding (size_any_t)` ：边界填充的长度。

`dilation (size_any_t)` ：在一次池化操作中，一次选择操作跨越多少像素。

`return_indices (bool)` ：是否返回池化后的值在原图像中的索引。

`ceil_mode (bool)` ：池化后是否将值向上取整。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.MaxPool1d(2) # [T, B, L] -> [T, B, L // 2]
```

## `matterhorn_pytorch.snn.MaxPool2d` / `matterhorn_pytorch.snn.layer.MaxPool2d`

二维最大池化层。将对脉冲的最大池化定义为：只要有任意一个输入产生脉冲，则输出产生脉冲。可以用如下公式描述：

$$O_{i}^{l}(t)=(\sum_{j=1}^{m}{O_{j}^{l-1}(t)} \ge 1)$$

其中 $i$ 为输出索引， $j$ 为输出索引为 $i$ 时应被统计的输入索引， $m$ 为应被统计的输入索引个数。脉冲的最大池化如下图所示。

![最大池化示意图](../../../assets/docs/snn/layer_1.png)

```python
MaxPool2d(
    kernel_size: _size_any_t,
    stride: Optional[_size_any_t] = None,
    padding: _size_any_t = 0,
    dilation: _size_any_t = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
    batch_first: bool = False
)
```

### 构造函数参数

`kernel_size (size_any_t)` ：池化核的大小。

`stride (size_any_t | None)` ：一次池化操作跨越多少像素。

`padding (size_any_t)` ：边界填充的长度。

`dilation (size_any_t)` ：在一次池化操作中，一次选择操作跨越多少像素。

`return_indices (bool)` ：是否返回池化后的值在原图像中的索引。

`ceil_mode (bool)` ：池化后是否将值向上取整。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.MaxPool2d(2) # [T, B, H, W] -> [T, B, H // 2, W // 2]
```

## `matterhorn_pytorch.snn.MaxPool3d` / `matterhorn_pytorch.snn.layer.MaxPool3d`

三维最大池化层。将对脉冲的最大池化定义为：只要有任意一个输入产生脉冲，则输出产生脉冲。可以用如下公式描述：

$$O_{i}^{l}(t)=(\sum_{j=1}^{m}{O_{j}^{l-1}(t)} \ge 1)$$

其中 $i$ 为输出索引， $j$ 为输出索引为 $i$ 时应被统计的输入索引， $m$ 为应被统计的输入索引个数。

```python
MaxPool3d(
    kernel_size: _size_any_t,
    stride: Optional[_size_any_t] = None,
    padding: _size_any_t = 0,
    dilation: _size_any_t = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
    batch_first: bool = False
)
```

### 构造函数参数

`kernel_size (size_any_t)` ：池化核的大小。

`stride (size_any_t | None)` ：一次池化操作跨越多少像素。

`padding (size_any_t)` ：边界填充的长度。

`dilation (size_any_t)` ：在一次池化操作中，一次选择操作跨越多少像素。

`return_indices (bool)` ：是否返回池化后的值在原图像中的索引。

`ceil_mode (bool)` ：池化后是否将值向上取整。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.MaxPool3d(2) # [T, B, H, W, L] -> [T, B, H // 2, W // 2, L // 2]
```

## `matterhorn_pytorch.snn.AvgPool1d` / `matterhorn_pytorch.snn.layer.AvgPool1d`

一维平均池化层。将对脉冲的平均池化定义为：当一半及以上的输入产生脉冲时，输出才产生脉冲。可以用如下公式描述：

$$O_{i}^{l}(t)=(\sum_{j=1}^{m}{O_{j}^{l-1}(t)} \ge \lceil \frac{m}{2} \rceil)$$

其中 $i$ 为输出索引， $j$ 为输出索引为 $i$ 时应被统计的输入索引， $m$ 为应被统计的输入索引个数。

```python
AvgPool1d(
    kernel_size: _size_1_t,
    stride: Optional[_size_1_t] = None,
    padding: _size_1_t = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    batch_first: bool = False
)
```

### 构造函数参数

`kernel_size (size_1_t)` ：池化核的大小。

`stride (size_1_t)` ：一次池化操作跨越多少像素。

`padding (size_1_t)` ：边界填充的长度。

`ceil_mode (bool)` ：池化后是否将值向上取整。

`count_include_pad (bool)` ：池化的时候是否连边界一起计入。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.AvgPool1d(2) # [T, B, L] -> [T, B, L // 2]
```

## `matterhorn_pytorch.snn.AvgPool2d` / `matterhorn_pytorch.snn.layer.AvgPool2d`

二维平均池化层。将对脉冲的平均池化定义为：当一半及以上的输入产生脉冲时，输出才产生脉冲。可以用如下公式描述：

$$O_{i}^{l}(t)=(\sum_{j=1}^{m}{O_{j}^{l-1}(t)} \ge \lceil \frac{m}{2} \rceil)$$

其中 $i$ 为输出索引， $j$ 为输出索引为 $i$ 时应被统计的输入索引， $m$ 为应被统计的输入索引个数。脉冲的平均池化如下图所示。

![平均池化示意图](../../../assets/docs/snn/layer_2.png)

```python
AvgPool2d(
    kernel_size: _size_2_t,
    stride: Optional[_size_2_t] = None,
    padding: _size_2_t = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
    batch_first: bool = False
)
```

### 构造函数参数

`kernel_size (size_2_t)` ：池化核的大小。

`stride (size_2_t | None)` ：一次池化操作跨越多少像素。

`padding (size_2_t)` ：边界填充的长度。

`ceil_mode (bool)` ：池化后是否将值向上取整。

`count_include_pad (bool)` ：池化的时候是否连边界一起计入。

`divisor_override (int | None)` ：是否用某个数取代总和作为除数。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.AvgPool2d(2) # [T, B, H, W] -> [T, B, H // 2, W // 2]
```

## `matterhorn_pytorch.snn.AvgPool3d` / `matterhorn_pytorch.snn.layer.AvgPool3d`

三维平均池化层。将对脉冲的平均池化定义为：当一半及以上的输入产生脉冲时，输出才产生脉冲。可以用如下公式描述：

$$O_{i}^{l}(t)=(\sum_{j=1}^{m}{O_{j}^{l-1}(t)} \ge \lceil \frac{m}{2} \rceil)$$

其中 $i$ 为输出索引， $j$ 为输出索引为 $i$ 时应被统计的输入索引， $m$ 为应被统计的输入索引个数。

```python
AvgPool3d(
    kernel_size: _size_3_t,
    stride: Optional[_size_3_t] = None,
    padding: _size_3_t = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
    batch_first: bool = False
)
```

### 构造函数参数

`kernel_size (size_3_t)` ：池化核的大小。

`stride (size_3_t | None)` ：一次池化操作跨越多少像素。

`padding (size_3_t)` ：边界填充的长度。

`ceil_mode (bool)` ：池化后是否将值向上取整。

`count_include_pad (bool)` ：池化的时候是否连边界一起计入。

`divisor_override (int | None)` ：是否用某个数取代总和作为除数。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.AvgPool3d(2) # [T, B, H, W, L] -> [T, B, H // 2, W // 2, L // 2]
```

## `matterhorn_pytorch.snn.MaxUnpool1d` / `matterhorn_pytorch.snn.layer.MaxUnpool1d`

一维最大反池化层。

```python
MaxUnpool1d(
    kernel_size: _Union[int, _Tuple[int]],
    stride: _Optional[_Union[int, _Tuple[int]]] = None,
    padding: _Union[int, _Tuple[int]] = 0,
    batch_first: bool = False
)
```

### 构造函数参数

`kernel_size (int*)` ：池化核的大小。

`stride (int* | None)` ：一次池化操作涉及多少像素。

`padding (int*)` ：边界填充的长度。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.MaxPool1d(2, return_indices = True) # [T, B, C, L] -> [T, B, C, L // 2]
up = mth.snn.MaxUnpool1d(2) # [T, B, C, L // 2] -> [T, B, C, L]

x = torch.rand(8, 2, 3, 16)
y, i = pooling(x)
print(y.shape)
z = up(y, i)
print(z.shape)
```

## `matterhorn_pytorch.snn.MaxUnpool2d` / `matterhorn_pytorch.snn.layer.MaxUnpool2d`

二维最大反池化层。

```python
MaxUnpool2d(
    kernel_size: _Union[int, _Tuple[int]],
    stride: _Optional[_Union[int, _Tuple[int]]] = None,
    padding: _Union[int, _Tuple[int]] = 0,
    batch_first: bool = False
)
```

### 构造函数参数

`kernel_size (int*)` ：池化核的大小。

`stride (int* | None)` ：一次池化操作涉及多少像素。

`padding (int*)` ：边界填充的长度。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.MaxPool2d(2, return_indices = True) # [T, B, C, H, W] -> [T, B, C, H // 2, W // 2]
up = mth.snn.MaxUnpool2d(2) # [T, B, C, H // 2, W // 2] -> [T, B, C, H, W]

x = torch.rand(8, 2, 3, 16, 16)
y, i = pooling(x)
print(y.shape)
z = up(y, i)
print(z.shape)
```

## `matterhorn_pytorch.snn.MaxUnpool3d` / `matterhorn_pytorch.snn.layer.MaxUnpool3d`

三维最大反池化层。

```python
MaxUnpool3d(
    kernel_size: _Union[int, _Tuple[int]],
    stride: _Optional[_Union[int, _Tuple[int]]] = None,
    padding: _Union[int, _Tuple[int]] = 0,
    batch_first: bool = False
)
```

### 构造函数参数

`kernel_size (int*)` ：池化核的大小。

`stride (int* | None)` ：一次池化操作涉及多少像素。

`padding (int*)` ：边界填充的长度。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.MaxPool3d(2, return_indices = True) # [T, B, C, H, W, L] -> [T, B, C, H // 2, W // 2, L // 2]
up = mth.snn.MaxUnpool3d(2) # [T, B, C, H // 2, W // 2, L // 2] -> [T, B, C, H, W, L]

x = torch.rand(8, 2, 3, 16, 16, 16)
y, i = pooling(x)
print(y.shape)
z = up(y, i)
print(z.shape)
```

## `matterhorn_pytorch.snn.Upsample` / `matterhorn_pytorch.snn.layer.Upsample`

上采样层，以某种方式进行上采样。

```python
Upsample(
    size: int | Tuple[int, int] | None = None,
    scale_factor: float | Tuple[float, float] | None = None,
    mode: str = 'nearest',
    align_corners: bool | None = None,
    recompute_scale_factor: bool | None = None,
    batch_first: bool = False
)
```

### 构造函数参数

`size (int | int*)` ：输出大小。在构造时与 `scale_factor` 选一个传入。

`scale_factor (float | float*)` ：比例因子，如 `2` 为上采样两倍。在构造时与 `size` 选一个传入。

`mode (str)` ：以何种形式上采样。可选参数为 `nearest`（最近邻），`linear`（单线性），`bilinear`（双线性），`bicubic`（两次立方），`trilinear`（三线性）。

`align_corners (bool)` ：若为 `True`，使输入和输出张量的角像素对齐，从而保留这些像素的值。

`recompute_scale_factor (bool)` ：若为 `True`，则必须传入 `scale_factor` 并且 `scale_factor` 用于计算输出大小。计算出的输出大小将用于推断插值的新比例；若为 `False`，那么 `size` 或 `scale_factor` 将直接用于插值。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


up = mth.snn.Upsample(scale_factor = 2.0, mode = "bilinear") # [T, B, H, W] -> [T, B, H * 2, W * 2]
x = torch.rand(8, 2, 3, 16, 16)
y = up(x)
print(y.shape)
```

## `matterhorn_pytorch.snn.Flatten` / `matterhorn_pytorch.snn.layer.Flatten`

展平层，将张量重排展开，一般用于连接卷积层与输出的全连接层。

```python
Flatten(
    start_dim: int = 1,
    end_dim: int = -1,
    batch_first: bool = False
)
```

### 构造函数参数

`start_dim (int)` ：展平开始的维度（不算时间维度）。默认为 `1` ，即跳过批维度，从空间维度开始展平。

`end_dim (int)` ：展平结束的维度（不算时间维度）。默认为 `-1` ，即展平到最后一个维度。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


lf = mth.snn.Flatten() # [T, B, H, W] -> [T, B, H * W]
```

## `matterhorn_pytorch.snn.Unflatten` / `matterhorn_pytorch.snn.layer.Unflatten`

反展平层，将展开的张量重新折叠。

```python
Unflatten(
    dim: Union[int, str],
    unflattened_size: _size,
    batch_first: bool = False
)
```

### 构造函数参数

`dim (int)` ：要折叠哪一个维度（不算时间维度）的数据。

`unflattened_size (size)` ：将这一个维度的数据折叠成什么形状。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


lf = mth.snn.Unflatten(1, (1, 28, 28)) # [T, B, 784] -> [T, B, 1, 28, 28]
```

## `matterhorn_pytorch.snn.Dropout` / `matterhorn_pytorch.snn.layer.Dropout`

遗忘层，以一定概率将元素置为 `0` 。

```python
Dropout(
    p: float = 0.5,
    inplace: bool = False,
    batch_first: bool = False
)
```

### 构造函数参数

`p (float)` ：遗忘概率。

`inplace (bool)` ：是否在原有张量上改动，若为 `True` 则直接改原张量，否则新建一个张量。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


ld = mth.snn.Dropout(0.5)
```

## `matterhorn_pytorch.snn.Dropout1d` / `matterhorn_pytorch.snn.layer.Dropout1d`

一维遗忘层，以一定概率将元素置为 `0` 。详情请参考 `matterhorn_pytorch.snn.Dropout` 。

## `matterhorn_pytorch.snn.Dropout2d` / `matterhorn_pytorch.snn.layer.Dropout2d`

二维遗忘层，以一定概率将元素置为 `0` 。详情请参考 `matterhorn_pytorch.snn.Dropout` 。

## `matterhorn_pytorch.snn.Dropout3d` / `matterhorn_pytorch.snn.layer.Dropout3d`

三维遗忘层，以一定概率将元素置为 `0` 。详情请参考 `matterhorn_pytorch.snn.Dropout` 。

## `matterhorn_pytorch.snn.TemporalWiseAttention` / `matterhorn_pytorch.snn.layer.TemporalWiseAttention`

逐时间注意力层。详情参见参考文献 [1]。

```python
TemporalWiseAttention(
    time_steps: int,
    d_threshold: float,
    expand: float = 1.0,
    batch_first: bool = False
)
```

### 构造函数参数

`time_steps (int)` ：时间步长 $T$。

`d_threshold (float)` ：注意力阈值 $\delta$。在推理时，小于这一阈值的时间步上的所有事件会被舍去。

`batch_first (bool)` ：第一个维度是批大小（`True`）还是时间步（`False`）。

### 示例用法

```python
import torch
import matterhorn_pytorch as mth


ta = mth.snn.TemporalWiseAttention(16, 0.9)
x = torch.rand(16, 3, 2, 8, 8)
y = ta(x)
print(y)
```

## 参考文献

[1] Yao M, Gao H, Zhao G, et al. Temporal-wise attention spiking neural networks for event streams classification[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 10221-10230.