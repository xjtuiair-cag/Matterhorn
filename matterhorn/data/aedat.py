import numpy as np
import torch
import torch.nn as nn
import os
from torchvision.datasets.utils import  check_integrity, download_url, extract_archive, verify_str_arg
from torch.utils.data import Dataset, DataLoader
from typing import Any, List, Tuple, Callable, Optional
from urllib.error import URLError
from rich import print


class AEDAT(Dataset):
    training_file = "training.pt"
    test_file = "test.pt"
    mirrors = []
    resources = []
    y_mask = 0x7F00
    y_shift = 8
    x_mask = 0x00FE
    x_shift = 1
    p_mask = 0x0001
    p_shift = 0
    
    
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, time_steps: int = 64, width: int = 128, height: int = 128, polarity: bool = True, endian: str = ">", datatype: str = "u4") -> None:
        """
        原始数据后缀名为.aedat的数据集
        @params:
            root: str 数据集的存储位置
            train: bool 是否为训练集
            transform: Callable | None 数据如何变换
            target_transform: Callable | None 标签如何变换
            download: bool 如果数据集不存在，是否应该下载
            time_steps: int 最终的数据集总共含有多少个时间步
            width: int 最终数据集的宽度
            height: int 最终数据集的高度
            polarity: bool 最终数据集是否采集极性信息，如果采集，通道数就是2，否则是1
            endian: str 大端还是小端，">"代表大端存储，"<"代表小端存储
            datatype: str 数据类型，如u4表示uint32
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.endian = endian
        self.datatype = datatype
        self.t_size = time_steps
        self.p_size = 2 if polarity else 1
        self.x_size = width
        self.y_size = height
        if self.check_legacy_exist():
            self.data, self.targets = self.load_legacy_data()
            return
        if download:
            self.download()
        if not self.check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        self.data, self.targets = self.load_data()


    @property
    def raw_folder(self) -> str:
        """
        刚下载下来的数据集所存储的地方
        @return:
            str 数据集存储位置
        """
        return os.path.join(self.root, self.__class__.__name__, "raw")


    @property
    def processed_folder(self) -> str:
        """
        处理过后的数据集所存储的地方
        @return:
            str 数据集存储位置
        """
        return os.path.join(self.root, self.__class__.__name__, "processed_t%d_p%d_h%d_w%d" % (self.t_size, self.p_size, self.y_size, self.x_size))

    
    def check_legacy_exist(self) -> bool:
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False
        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )


    def load_legacy_data(self) -> Any:
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))
    
    
    def check_exists(self) -> bool:
        """
        检查是否存在
        @return:
            if_exist: bool 是否存在
        """
        return False
        
    
    def download(self) -> None:
        """
        下载数据集
        """
        pass


    def load_data(self) -> Any:
        """
        预处理数据集
        """
        pass

    
    def extract(self, data: np.ndarray, mask: int, shift: int) -> np.ndarray:
        """
        从事件数据中提取x,y,p值所用的函数
        @params:
            data: np.ndarray 事件数据
            mask: int 对应的掩模
            shift: int 对应的偏移量
        @return:
            data: np.ndarray 处理后的数据（x,y,p）
        """
        return (data & mask) >> shift
        
    
    def filename_2_data(self, filename: str) -> np.ndarray:
        """
        输入文件名，读取文件内容
        @params:
            filename: str 文件名
        @return:
            data: np.ndarray 文件内容（数据）
        """
        data_str = ""
        with open(filename, 'rb') as f:
            data_str = f.read()
            lines = data_str.split(b'\n')
            for line in range(len(lines)):
                if not lines[line].startswith(b'#'):
                    break
            lines = lines[line:]
            data_str = b'\n'.join(lines)
        data = np.fromstring(data_str, dtype = self.endian + self.datatype)
        return data
    
    
    def data_2_tpyx(self, data: np.ndarray) -> np.ndarray:
        """
        将数据分割为t,x,y,p数组
        @params:
            data: np.ndarray 数据，形状为[2n]
        @return:
            data_tpyx: np.ndarray 分为t,p,y,x的数据，形状为[n, 4]
        """
        res = np.zeros(data.shape[0] // 2, 4)
        xyp = data[::2]
        t = data[1::2]
        res[0] = t
        res[1] = self.extract(xyp, self.x_mask, self.x_shift)
        res[2] = self.extract(xyp, self.y_mask, self.y_shift)
        res[3] = self.extract(xyp, self.p_mask, self.p_shift)
        return res
    
    
    def tpyx_2_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        将t,p,y,x数组转为最后的PyTorch张量
        @params:
            data: np.ndarray 数据，形状为[n, 4]
        @return:
            data_tensor: torch.Tensor 渲染成事件的张量，形状为[T, C(P), H, W]
        """
        pass
    
    
    def label(self, path: str) -> int:
        """
        返回该文件对应的标签
        @params:
            path: str 文件的存储路径
        @return:
            label: int 文件的标签
        """
        pass
    
    
    def is_train(self, path: str, index: int = 0) -> bool:
        """
        从路径、文件名和索引判断是否是训练集
        @params:
            path: str 文件的存储路径
            index: int 文件的索引
        @return:
            is_train: bool 是否为训练集
        """
        pass
    
    
    def __sizeof__(self) -> int:
        """
        获取数据集长度
        @return:
            len: int 数据集长度
        """
        return super().__sizeof__()
    
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据集
        @params:
            index: int 索引
        @return:
            x: torch.Tensor 数据
            y: torch.Tensor 标签
        """
        return super().__getitem__(index)


class CIFAR10DVS(AEDAT):
    mirrors = ["https://ndownloader.figshare.com/files/"]
    resources = [
        ("7712788", "airplane.zip", "0afd5c4bf9ae06af762a77b180354fdd"),
        ("7712791", "automobile.zip", "8438dfeba3bc970c94962d995b1b9bdd"),
        ("7712794", "bird.zip", "a9c207c91c55b9dc2002dc21c684d785"),
        ("7712812", "cat.zip", "52c63c677c2b15fa5146a8daf4d56687"),
        ("7712815", "deer.zip", "b6bf21f6c04d21ba4e23fc3e36c8a4a3"),
        ("7712818", "dog.zip", "f379ebdf6703d16e0a690782e62639c3"),
        ("7712842", "frog.zip", "cad6ed91214b1c7388a5f6ee56d08803"),
        ("7712851", "horse.zip", "e7cbbf77bec584ffbf913f00e682782a"),
        ("7712836", "ship.zip", "41c7bd7d6b251be82557c6cce9a7d5c9"),
        ("7712839", "truck.zip", "89f3922fd147d9aeff89e76a2b0b70a7")
    ]


    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, time_steps: int = 64, width: int = 128, height: int = 128, polarity: bool = True) -> None:
        """
        CIFAR-10 DVS数据集，将CIFAR10数据集投影至LCD屏幕后，用事件相机录制的数据集
        @params:
            root: str 数据集的存储位置
            train: bool 是否为训练集
            transform: Callable | None 数据如何变换
            target_transform: Callable | None 标签如何变换
            download: bool 如果数据集不存在，是否应该下载
            width: int 最终数据集的宽度
            height: int 最终数据集的高度
            polarity: bool 最终数据集是否采集极性信息，如果采集，通道数就是2，否则是1
        """
        super().__init__(
            root = root,
            train = train,
            transform = transform,
            target_transform = target_transform,
            download = download,
            time_steps = time_steps,
            width = width,
            height = height,
            polarity = polarity,
            endian = ">",
            datatype = "u4"
        )


    def check_exists(self) -> bool:
        """
        检查是否存在
        @return:
            if_exist: bool 是否存在
        """
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(filename))[0]))
            for fileurl, filename, md5 in self.resources
        )


    def download(self) -> None:
        """
        下载CIFAR-10 DVS数据集
        """
        if self.check_exists():
            return
        os.makedirs(self.raw_folder, exist_ok = True)
        for fileurl, filename, md5 in self.resources:
            if os.path.isfile(self.raw_folder + os.sep + filename):
                print("[blue]File %s%s%s has already existed.[/blue]" % (self.raw_folder, os.sep, filename))
                continue
            is_downloaded = False
            for mirror in self.mirrors:
                url = mirror + fileurl
                try:
                    print("[blue]Downloading %s%s%s from %s[/blue]" % (self.raw_folder, os.sep, filename, url))
                    download_url(url, root = self.raw_folder, filename = filename, md5 = md5)
                    is_downloaded = True
                    break
                except URLError as error:
                    print("[red]Error in file %s%s%s downloaded from %s:\r\n\r\n%s\r\n\r\nPlease manually download it.[/red]" % (self.raw_folder, os.sep, filename, url, error))
                    is_downloaded = False
            if is_downloaded:
                print("[green]Successfully downloaded %s%s%s[/green]" % (self.raw_folder, os.sep, filename))
    
    
    def load_data(self) -> Any:
        return
