# `matterhorn_pytorch.data`

[Back to `matterhorn_pytorch`](../README.md)

[English](../../en_us/data/README.md)

[中文](../../zh_cn/data/README.md)

## Module Introduction

The module for neuromorphic datasets in Matterhorn, containing handling methods for various neuromorphic datasets.

## `matterhorn_pytorch.data.CIFAR10DVS` / `matterhorn_pytorch.data.aedat.CIFAR10DVS`

CIFAR10-DVS neuromorphic dataset.

```python
CIFAR10DVS(
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    cached: bool = True,
    count: bool = False,
    time_steps: int = 128,
    width: int = 128,
    height: int = 128,
    polarity: bool = True
)
```

### Constructor Arguments

`root (str)`: Location where all the dataset data is stored.

`train (bool)`: Whether it is the training set. If `True`, loads the training set data; if `False`, loads the test set data.

`transform (Callable | None)`: Transformation applied to the data in the dataset.

`target_transform (Callable | None)`: Transformation applied to the labels in the dataset.

`download (bool)`: Whether to attempt downloading the dataset resources if not available locally.

`cached (bool)`: Whether to use cache. Cache will make costs on disk space but save the time used for training and validating.

`cache_dtype (torch.dtype)` ：If using cache, this parameter indicates the data type of cache. Default is `torch.uint8`. Set it to `torch.float` to use higher precision to save the count data when the parameter `count = True`.

`count (bool)`: Whether to output the event count for each point as spike output. If `False`, the spike sequence will only have `0` and `1`; otherwise, the output is the event count of that point.

`time_steps (int)`: Time steps `T` for the output tensor.

`width (int)`: Width `W` of the output tensor.

`height (int)`: Height `H` of the output tensor.

`polarity (bool)`: Whether the output tensor contains polarity. If `True`, the output tensor contains polarity, `P = 2`; otherwise, it does not, `P = 1`.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth
from matterhorn_pytorch.data import CIFAR10DVS


test_dataset = CIFAR10DVS(
    root = './examples/data',
    train = False,
    download = True,
    time_steps = 128
)
demo_data, demo_label = test_dataset[0]
mth.util.plotter.event_plot_tyx(demo_data, titles = ["Label %s" % test_dataset.labels[demo_label]])
```

![Dataset Example](../../../assets/docs/data/readme_1.png)

## `matterhorn_pytorch.data.DVS128Gesture` / `matterhorn_pytorch.data.aedat.DVS128Gesture`

DVS128 Gesture neuromorphic dataset.

**Note**: This dataset cannot be downloaded even if you set `download` to `True`. You should download all the data from [https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794) and place it into the `root/raw` folder.

```python
DVS128Gesture(
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    cached: bool = True,
    count: bool = False,
    time_steps: int = 128,
    width: int = 128,
    height: int = 128,
    polarity: bool = True
)
```

### Constructor Arguments

`root (str)`: Location where all the dataset data is stored.

`train (bool)`: Whether it is the training set. If `True`, loads the training set data; if `False`, loads the test set data.

`transform (Callable | None)`: Transformation applied to the data in the dataset.

`target_transform (Callable | None)`: Transformation applied to the labels in the dataset.

`download (bool)`: Whether to attempt downloading the dataset resources if not available locally.

`cached (bool)`: Whether to use cache. Cache will make costs on disk space but save the time used for training and validating.

`cache_dtype (torch.dtype)` ：If using cache, this parameter indicates the data type of cache. Default is `torch.uint8`. Set it to `torch.float` to use higher precision to save the count data when the parameter `count = True`.

`count (bool)`: Whether to output the event count for each point as spike output. If `False`, the spike sequence will only have `0` and `1`; otherwise, the output is the event count of that point.

`time_steps (int)`: Time steps `T` for the output tensor.

`width (int)`: Width `W` of the output tensor.

`height (int)`: Height `H` of the output tensor.

`polarity (bool)`: Whether the output tensor contains polarity. If `True`, the output tensor contains polarity, `P = 2`; otherwise, it does not, `P = 1`.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth
from matterhorn_pytorch.data import DVS128Gesture


test_dataset = DVS128Gesture(
    root = './examples/data',
    train = False,
    download = True,
    time_steps = 128
)
demo_data, demo_label = test_dataset[0]
mth.util.plotter.event_plot_tyx(demo_data, titles = ["Label %s" % test_dataset.labels[demo_label]])
```

![Dataset Example](../../../assets/docs/data/readme_2.png)

## `matterhorn_pytorch.data.SpikingHeidelbergDigits` / `matterhorn_pytorch.data.hdf5.SpikingHeidelbergDigits`

Spiking Heidelberg Digits (SHD) neuromorphic dataset.

```python
CIFAR10DVS(
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    cached: bool = True,
    count: bool = False,
    precision: float = 1e9,
    time_steps: int = 128,
    length: int = 700
)
```

### Constructor Arguments

`root (str)`: Location where all the dataset data is stored.

`train (bool)`: Whether it is the training set. If `True`, loads the training set data; if `False`, loads the test set data.

`transform (Callable | None)`: Transformation applied to the data in the dataset.

`target_transform (Callable | None)`: Transformation applied to the labels in the dataset.

`download (bool)`: Whether to attempt downloading the dataset resources if not available locally.

`cached (bool)`: Whether to use cache. Cache will make costs on disk space but save the time used for training and validating.

`cache_dtype (torch.dtype)` ：If using cache, this parameter indicates the data type of cache. Default is `torch.uint8`. Set it to `torch.float` to use higher precision to save the count data when the parameter `count = True`.

`count (bool)`: Whether to output the event count for each point as spike output. If `False`, the spike sequence will only have `0` and `1`; otherwise, the output is the event count of that point.

`precision (float)`: Precision of the dataset. Keep the default value `1e9`.

`time_steps (int)`: Time steps `T` for the output tensor.

`length (int)`: Spatial length `L` of the output tensor.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth
from matterhorn_pytorch.data import SpikingHeidelbergDigits


test_dataset = SpikingHeidelbergDigits(
    root = './examples/data',
    train = False,
    download = True,
    time_steps = 128
)
demo_data, demo_label = test_dataset[0]
mth.util.plotter.event_plot_tx(demo_data, titles = ["Label %s" % test_dataset.labels[demo_label]])
```

![Dataset Example](../../../assets/docs/data/readme_3.png)

## `matterhorn_pytorch.data.NMNIST` / `matterhorn_pytorch.data.nmnist.NMNIST`

N-MNIST neuromorphic dataset.

```python
NMNIST(
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    cached: bool = True,
    count: bool = False,
    time_steps: int = 128,
    width: int = 34,
    height: int = 34,
    polarity: bool = True
)
```

### Constructor Arguments

`root (str)`: Location where all the dataset data is stored.

`train (bool)`: Whether it is the training set. If `True`, loads the training set data; if `False`, loads the test set data.

`transform (Callable | None)`: Transformation applied to the data in the dataset.

`target_transform (Callable | None)`: Transformation applied to the labels in the dataset.

`download (bool)`: Whether to attempt downloading the dataset resources if not available locally.

`cached (bool)`: Whether to use cache. Cache will make costs on disk space but save the time used for training and validating.

`cache_dtype (torch.dtype)` ：If using cache, this parameter indicates the data type of cache. Default is `torch.uint8`. Set it to `torch.float` to use higher precision to save the count data when the parameter `count = True`.

`count (bool)`: Whether to output the event count for each point as spike output. If `False`, the spike sequence will only have `0` and `1`; otherwise, the output is the event count of that point.

`time_steps (int)`: Time steps `T` for the output tensor.

`width (int)`: Width `W` of the output tensor.

`height (int)`: Height `H` of the output tensor.

`polarity (bool)`: Whether the output tensor contains polarity. If `True`, the output tensor contains polarity, `P = 2`; otherwise, it does not, `P = 1`.

### Example Usage

```python
import torch
import matterhorn_pytorch as mth
from matterhorn_pytorch.data import NMNIST


test_dataset = NMNIST(
    root = './examples/data',
    train = False,
    download = True,
    time_steps = 128
)
demo_data, demo_label = test_dataset[0]
mth.util.plotter.event_plot_tyx(demo_data, titles = ["Label %s" % test_dataset.labels[demo_label]])
```

![Dataset Example](../../../assets/docs/data/readme_4.png)