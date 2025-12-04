# `matterhorn_pytorch.snn.layer`

[Back to `matterhorn_pytorch.snn`](./README.md)

[English](../../en_us/snn/6_layer.md)

[中文](../../zh_cn/snn/6_layer.md)

## Module Introduction

This module defines some complete neural network layers or inter-layer operations, such as pooling, flattening, etc.

## `matterhorn_pytorch.snn.layer.Layer`

```python
Layer(
    batch_first: bool = False
)
```

### Constructor Arguments

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Overridable Methods

#### `forward(self, o: torch.Tensor) -> torch.Tensor`

Same as `nn.Module.forward`.

## `matterhorn_pytorch.snn.STDPLinear` / `matterhorn_pytorch.snn.layer.STDPLinear`

A fully connected layer employing the Spike Timing-Dependent Plasticity (STDP) learning mechanism. STDP, as a common spike-based learning mechanism, follows the weight update formula:

$$\Delta w_{ij}=\sum_{t_{j}}{\sum_{t_{i}}W(t_{i}-t_{j})}$$

Where the weight function $W(x)$ is defined as:

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

Where $t_{i}$ is the time step when the input spike of neuron with index $i$ is generated, and $t_{j}$ is the time step when the input spike of the synapse with index $j$ arrives.

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

### Constructor Arguments

`soma (torch.nn.Module)`: Type of soma used. Refer to the available soma types in the module [`matterhorn_pytorch.snn.soma`](./4_soma.md).

`in_features (int)`: Length of the input `I`. The shape of the input is `[B, I]` (single-time-step mode) or `[T, B, I]` (multi-time-step mode).

`out_features (int)`: Length of the output `O`. The shape of the output is `[B, O]` (single-time-step mode) or `[T, B, O]` (multi-time-step mode).

`a_pos (float)`: STDP parameter $A_{+}$.

`tau_pos (float)`: STDP parameter $\tau_{+}$.

`a_neg (float)`: STDP parameter $A_{-}$.

`tau_neg (float)`: STDP parameter $\tau_{-}$.

`device (torch.device)`: Computational device used.

`dtype (torch.dtype)`: Data type used for computation.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


l1 = mth.snn.STDPLinear(mth.snn.LIF(), 784, 10) # [T, B, 784] -> [T, B, 10]
```

## `matterhorn_pytorch.snn.STDPConv2d` / `matterhorn_pytorch.snn.layer.STDPConv2d`

A 2D convolutional layer employing the Spike-Timing-Dependent Plasticity (STDP) learning mechanism.

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

### Constructor Parameters

`soma (torch.nn.Module)`: The type of soma to use. Available soma types can be found in the module [`matterhorn_pytorch.snn.soma`](./4_soma.md).

`in_channels (int)`: Number of input channels `CI`. The input shape is `[B, CI, HI, WI]` (single-time-step mode) or `[T, B, CI, HI, WI]` (multi-time-step mode).

`out_channels (int)`: Number of output channels `CO`. The output shape is `[B, CO, HO, WO]` (single-time-step mode) or `[T, B, CO, HO, WO]` (multi-time-step mode).

`kernel_size (size_2_t)`: Shape of the convolution kernel.

`stride (size_2_t)`: Stride. The step size for moving the kernel across the input.

`padding (size_2_t | str)`: Padding size. The amount of padding added to the borders.

`dilation (size_2_t)`: Spacing between kernel elements. A value of 1 corresponds to standard convolution.

`a_pos (float)`: STDP parameter $A_{+}$.

`tau_pos (float)`: STDP parameter $\tau_{+}$.

`a_neg (float)`: STDP parameter $A_{-}$.

`tau_neg (float)`: STDP parameter $\tau_{-}$.

`device (torch.device)`: The computing device used for calculation.

`dtype (torch.dtype)`: The data type used for calculation.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


l1 = mth.snn.STDPConv2d(mth.snn.LIF(), 2, 32, 3, 2, 1) # [T, B, 2, 32, 32] -> [T, B, 32, 16, 16]
```

## `matterhorn_pytorch.snn.MaxPool1d` / `matterhorn_pytorch.snn.layer.MaxPool1d`

One-dimensional max pooling layer. Define max pooling of spikes as: as long as any input generates a spike, the output generates a spike. It can be described by the following formula:

$$O_{i}^{l}(t)=(\sum_{j=1}^{m}{O_{j}^{l-1}(t)} \ge 1)$$

Where $i$ is the output index, $j$ is the input index that should be counted when the output index is $i$, and $m$ is the number of input indices to be counted.

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

### Constructor Arguments

`kernel_size (size_any_t)`: Size of the pooling kernel.

`stride (size_any_t | None)`: Number of pixels the pooling operation jumps over at once.

`padding (size_any_t)`: Length of padding at the boundaries.

`dilation (size_any_t)`: Number of pixels the pooling operation selects at once in one pooling operation.

`return_indices (bool)`: Whether to return the indices of the pooled values in the original image.

`ceil_mode (bool)`: Whether to ceil the value after pooling.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.MaxPool1d(2) # [T, B, L] -> [T, B, L // 2]
```

## `matterhorn_pytorch.snn.MaxPool2d` / `matterhorn_pytorch.snn.layer.MaxPool2d`

Two-dimensional max pooling layer. Define max pooling of spikes as: as long as any input generates a spike, the output generates a spike. It can be described by the following formula:

$$O_{i}^{l}(t)=(\sum_{j=1}^{m}{O_{j}^{l-1}(t)} \ge 1)$$

Where $i$ is the output index, $j$ is the input index that should be counted when the output index is $i$, and $m$ is the number of input indices to be counted. Max pooling of spikes is illustrated in the following figure.

![Max Pooling Illustration](../../../assets/docs/snn/layer_1.png)

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

### Constructor Arguments

`kernel_size (size_any_t)`: Size of the pooling kernel.

`stride (size_any_t | None)`: Number of pixels the pooling operation jumps over at once.

`padding (size_any_t)`: Length of padding at the boundaries.

`dilation (size_any_t)`: Number of pixels the pooling operation selects at once in one pooling operation.

`return_indices (bool)`: Whether to return the indices of the pooled values in the original image.

`ceil_mode (bool)`: Whether to ceil the value after pooling.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.MaxPool2d(2) # [T, B, H, W] -> [T, B, H // 2, W // 2]
```

## `matterhorn_pytorch.snn.MaxPool3d` / `matterhorn_pytorch.snn.layer.MaxPool3d`

Three-dimensional max pooling layer. Define max pooling of spikes as: as long as any input generates a spike, the output generates a spike. It can be described by the following formula:

$$O_{i}^{l}(t)=(\sum_{j=1}^{m}{O_{j}^{l-1}(t)} \ge 1)$$

Where $i$ is the output index, $j$ is the input index that should be counted when the output index is $i$, and $m$ is the number of input indices to be counted.

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

### Constructor Arguments

`kernel_size (size_any_t)`: Size of the pooling kernel.

`stride (size_any_t | None)`: Number of pixels the pooling operation jumps over at once.

`padding (size_any_t)`: Length of padding at the boundaries.

`dilation (size_any_t)`: Number of pixels the pooling operation selects at once in one pooling operation.

`return_indices (bool)`: Whether to return the indices of the pooled values in the original image.

`ceil_mode (bool)`: Whether to ceil the value after pooling.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.MaxPool3d(2) # [T, B, H, W, L] -> [T, B, H // 2, W // 2, L // 2]
```

## `matterhorn_pytorch.snn.AvgPool1d` / `matterhorn_pytorch.snn.layer.AvgPool1d`

One-dimensional average pooling layer. Define average pooling of spikes as: when half or more of the inputs generate spikes, the output generates spikes. It can be described by the following formula:

$$O_{i}^{l}(t)=(\sum_{j=1}^{m}{O_{j}^{l-1}(t)} \ge \lceil \frac{m}{2} \rceil)$$

Where $i$ is the output index, $j$ is the input index that should be counted when the output index is $i$, and $m$ is the number of input indices to be counted.

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

### Constructor Arguments

`kernel_size (size_1_t)`: Size of the pooling kernel.

`stride (size_1_t)`: Number of pixels the pooling operation jumps over at once.

`padding (size_1_t)`: Length of padding at the boundaries.

`ceil_mode (bool)`: Whether to ceil the value after pooling.

`count_include_pad (bool)`: Whether to include the boundary when pooling.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.AvgPool1d(2) # [T, B, L] -> [T, B, L // 2]
```

## `matterhorn_pytorch.snn.AvgPool2d` / `matterhorn_pytorch.snn.layer.AvgPool2d`

Two-dimensional average pooling layer. Define average pooling of spikes as: when half or more of the inputs generate spikes, the output generates spikes. It can be described by the following formula:

$$O_{i}^{l}(t)=(\sum_{j=1}^{m}{O_{j}^{l-1}(t)} \ge \lceil \frac{m}{2} \rceil)$$

Where $i$ is the output index, $j$ is the input index that should be counted when the output index is $i$, and $m$ is the number of input indices to be counted. Average pooling of spikes is illustrated in the following figure.

![Average Pooling Illustration](../../../assets/docs/snn/layer_2.png)

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

### Constructor Arguments

`kernel_size (size_2_t)`: Size of the pooling kernel.

`stride (size_2_t | None)`: Number of pixels the pooling operation jumps over at once.

`padding (size_2_t)`: Length of padding at the boundaries.

`ceil_mode (bool)`: Whether to ceil the value after pooling.

`count_include_pad (bool)`: Whether to include the boundary when pooling.

`divisor_override (int | None)`: Whether to use a specific number to replace the sum as the divisor.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.AvgPool2d(2) # [T, B, H, W] -> [T, B, H // 2, W // 2]
```

## `matterhorn_pytorch.snn.AvgPool3d` / `matterhorn_pytorch.snn.layer.AvgPool3d`

Three-dimensional average pooling layer. Define average pooling of spikes as: when half or more of the inputs generate spikes, the output generates spikes. It can be described by the following formula:

$$O_{i}^{l}(t)=(\sum_{j=1}^{m}{O_{j}^{l-1}(t)} \ge \lceil \frac{m}{2} \rceil)$$

Where $i$ is the output index, $j$ is the input index that should be counted when the output index is $i$, and $m$ is the number of input indices to be counted.

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

### Constructor Arguments

`kernel_size (size_3_t)`: Size of the pooling kernel.

`stride (size_3_t | None)`: Number of pixels the pooling operation jumps over at once.

`padding (size_3_t)`: Length of padding at the boundaries.

`ceil_mode (bool)`: Whether to ceil the value after pooling.

`count_include_pad (bool)`: Whether to include the boundary when pooling.

`divisor_override (int | None)`: Whether to use a specific number to replace the sum as the divisor.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


pooling = mth.snn.AvgPool3d(2) # [T, B, H, W, L] -> [T, B, H // 2, W // 2, L // 2]
```

## `matterhorn_pytorch.snn.MaxUnpool1d` / `matterhorn_pytorch.snn.layer.MaxUnpool1d`

1D maximum unpooling layer.

```python
MaxUnpool1d(
    kernel_size: _Union[int, _Tuple[int]],
    stride: _Optional[_Union[int, _Tuple[int]]] = None,
    padding: _Union[int, _Tuple[int]] = 0,
    batch_first: bool = False
)
```

### Constructor Parameters

`kernel_size (int*)`: Size of the pooling kernel.

`stride (int* | None)`: The step size for moving the pooling window. If `None`, it defaults to `kernel_size`.

`padding (int*)`: The amount of implicit zero padding that was added to the input during the corresponding max pooling operation.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

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

2D maximum unpooling layer.

```python
MaxUnpool2d(
    kernel_size: _Union[int, _Tuple[int]],
    stride: _Optional[_Union[int, _Tuple[int]]] = None,
    padding: _Union[int, _Tuple[int]] = 0,
    batch_first: bool = False
)
```

### Constructor Parameters

`kernel_size (int*)`: Size of the pooling kernel.

`stride (int* | None)`: The step size for moving the pooling window. If `None`, it defaults to `kernel_size`.

`padding (int*)`: The amount of implicit zero padding that was added to the input during the corresponding max pooling operation.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

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

3D maximum unpooling layer.

```python
MaxUnpool3d(
    kernel_size: _Union[int, _Tuple[int]],
    stride: _Optional[_Union[int, _Tuple[int]]] = None,
    padding: _Union[int, _Tuple[int]] = 0,
    batch_first: bool = False
)
```

### Constructor Parameters

`kernel_size (int*)`: Size of the pooling kernel.

`stride (int* | None)`: The step size for moving the pooling window. If `None`, it defaults to `kernel_size`.

`padding (int*)`: The amount of implicit zero padding that was added to the input during the corresponding max pooling operation.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

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

Upsampling layer, which upsamples the input using a specified method.

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

### Constructor Parameters

`size (int | int*)`: The target output size. Provide either this or `scale_factor` during construction.

`scale_factor (float | float*)`: The multiplier for the spatial size. For example, `2` means upsampling by a factor of 2. Provide either this or `size` during construction.

`mode (str)`: The upsampling algorithm to use. Options are: `nearest`, `linear`, `bilinear`, `bicubic`, `trilinear`.

`align_corners (bool)`: If `True`, the corner pixels of the input and output tensors are aligned, preserving their values.

`recompute_scale_factor (bool)`: If `True`, `scale_factor` must be provided and is used to compute the output `size`. The computed output size will be used to infer new scales for the interpolation. If `False`, `size` or `scale_factor` will be used directly for interpolation.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


up = mth.snn.Upsample(scale_factor = 2.0, mode = "bilinear") # [T, B, H, W] -> [T, B, H * 2, W * 2]
x = torch.rand(8, 2, 3, 16, 16)
y = up(x)
print(y.shape)
```

## `matterhorn_pytorch.snn.Flatten` / `matterhorn_pytorch.snn.layer.Flatten`

Flattening layer, reshapes and flattens the tensor. Generally used to connect convolutional layers with output fully connected layers.

```python
Flatten(
    start_dim: int = 1,
    end_dim: int = -1,
    batch_first: bool = False
)
```

### Constructor Arguments

`start_dim (int)`: Dimension to start flattening (excluding the time dimension). Default is `1`, starting from spatial dimensions.

`end_dim (int)`: Dimension to end flattening (excluding the time dimension). Default is `-1`, flattening to the last dimension.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


lf = mth.snn.Flatten() # [T, B, H, W] -> [T, B, H * W]
```

## `matterhorn_pytorch.snn.Unflatten` / `matterhorn_pytorch.snn.layer.Unflatten`

Unflattening layer, re-folds the flattened tensor.

```python
Unflatten(
    dim: Union[int, str],
    unflattened_size: _size,
    batch_first: bool = False
)
```

### Constructor Arguments

`dim (int)`: Which dimension (excluding the time dimension) of data to fold.

`unflattened_size (size)`: Shape to fold this dimension into.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


lf = mth.snn.Unflatten(1, (1, 28, 28)) # [T, B, 784] -> [T, B, 1, 28, 28]
```

## `matterhorn_pytorch.snn.Dropout` / `matterhorn_pytorch.snn.layer.Dropout`

Dropout layer, which sets elements to `0` with a certain probability.

```python
Dropout(
    p: float = 0.5,
    inplace: bool = False,
    batch_first: bool = False
)
```

### Constructor Arguments

`p (float)`: The dropout probability.

`inplace (bool)`: Whether to modify the original tensor. If `True`, modifies the original tensor; otherwise, creates a new tensor.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


ld = mth.snn.Dropout(0.5)
```

## `matterhorn_pytorch.snn.Dropout1d` / `matterhorn_pytorch.snn.layer.Dropout1d`

One-dimensional dropout layer, which sets elements to `0` with a certain probability. Refer to `matterhorn_pytorch.snn.Dropout` for more information.

## `matterhorn_pytorch.snn.Dropout2d` / `matterhorn_pytorch.snn.layer.Dropout2d`

Two-dimensional dropout layer, which sets elements to `0` with a certain probability. Refer to `matterhorn_pytorch.snn.Dropout` for more information.

## `matterhorn_pytorch.snn.Dropout3d` / `matterhorn_pytorch.snn.layer.Dropout3d`

Three-dimensional dropout layer, which sets elements to `0` with a certain probability. Refer to `matterhorn_pytorch.snn.Dropout` for more information.

## `matterhorn_pytorch.snn.TemporalWiseAttention` / `matterhorn_pytorch.snn.layer.TemporalWiseAttention`

Temporal-wise attention layer. For details, see reference [1].

```python
TemporalWiseAttention(
    time_steps: int,
    d_threshold: float,
    expand: float = 1.0,
    batch_first: bool = False
)
```

### Constructor Parameters

`time_steps (int)`: Time step length $T$.

`d_threshold (float)`: Attention threshold $\delta$. During inference, all events on time steps below this threshold are discarded.

`batch_first (bool)`: Whether the first dimension is batch size (`True`) or time step (`False`).

### Example Usage

```python
import torch
import matterhorn_pytorch as mth


ta = mth.snn.TemporalWiseAttention(16, 0.9)
x = torch.rand(16, 3, 2, 8, 8)
y = ta(x)
print(y)
```

## References

[1] Yao M, Gao H, Zhao G, et al. Temporal-wise attention spiking neural networks for event streams classification[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 10221-10230.