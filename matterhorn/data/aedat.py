from zipfile import BadZipFile
import numpy as np
import torch
import torch.nn as nn
import os
import random
from torchvision.datasets.utils import  check_integrity, download_url, extract_archive, verify_str_arg
from torch.utils.data import Dataset, DataLoader
from typing import Any, List, Tuple, Union, Callable, Optional
from urllib.error import URLError
from rich import print
from rich.progress import track


class AEDAT(Dataset):
    training_file = "training.pt"
    test_file = "test.pt"
    original_size = (0, 2, 128, 128)
    mirrors = []
    resources = []
    labels = []
    y_mask = 0x07FF0000
    y_shift = 16
    x_mask = 0x0000FFFE
    x_shift = 1
    p_mask = 0x00000001
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
        if download:
            self.download()
        if not self.check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        self.data_target = self.load_data()
        self.data_target = self.data_target[self.data_target[:, 2] == (1 if self.train else 0)][:, :2]
        self.data_target = self.data_target.tolist()
        random.shuffle(self.data_target)


    @property
    def raw_folder(self) -> str:
        """
        刚下载下来的数据集所存储的地方。
        @return:
            str 数据集存储位置
        """
        return os.path.join(self.root, self.__class__.__name__, "raw")


    @property
    def extracted_folder(self) -> str:
        """
        解压过后的数据集所存储的地方。
        @return:
            str 数据集存储位置
        """
        return os.path.join(self.root, self.__class__.__name__, "extracted")


    @property
    def processed_folder(self) -> str:
        """
        处理过后的数据集所存储的地方。
        @return:
            str 数据集存储位置
        """
        return os.path.join(self.root, self.__class__.__name__, "processed")
    
    
    def check_exists(self) -> bool:
        """
        检查是否存在。
        @return:
            if_exist: bool 是否存在
        """
        return False
        
    
    def download(self) -> None:
        """
        下载数据集。
        """
        return


    def load_data(self) -> np.ndarray:
        """
        加载数据集。
        @return:
            data_label: np.ndarray 数据信息，包括3列：数据集、标签、其为训练集（1）还是测试集（0）。
        """
        return None

    
    def extract(self, data: np.ndarray, mask: int, shift: int) -> np.ndarray:
        """
        从事件数据中提取x,y,p值所用的函数。
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
        输入文件名，读取文件内容。
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
        将数据分割为t,p,y,x数组。
        @params:
            data: np.ndarray 数据，形状为[2n]
        @return:
            data_tpyx: np.ndarray 分为t,p,y,x的数据，形状为[n, 4]
        """
        res = np.zeros((data.shape[0] // 2, 4), dtype = np.int)
        xyp = data[::2]
        t = data[1::2]
        res[:, 0] = t
        res[:, 1] = self.extract(xyp, self.p_mask, self.p_shift)
        res[:, 2] = self.extract(xyp, self.y_mask, self.y_shift)
        res[:, 3] = self.extract(xyp, self.x_mask, self.x_shift)
        return res
    
    
    def tpyx_2_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        将t,p,y,x数组转为最后的PyTorch张量。
        @params:
            data: np.ndarray 数据，形状为[n, 4]
        @return:
            data_tensor: torch.Tensor 渲染成事件的张量，形状为[T, C(P), H, W]
        """
        return torch.tensor(data)
    
    
    def label(self, key: str) -> int:
        """
        返回该文件对应的标签。
        @params:
            key: str 关键词
        @return:
            label: int 文件的标签
        """
        if key in self.labels:
            return self.labels.index(key)
        return -1
    
    
    def is_train(self, label: int, index: int = 0) -> bool:
        """
        从路径、文件名和索引判断是否是训练集。
        @params:
            label: int 标签
            index: int 文件的索引
        @return:
            is_train: bool 是否为训练集
        """
        return index < 900
    
    
    def __len__(self) -> int:
        """
        获取数据集长度。
        @return:
            len: int 数据集长度
        """
        return len(self.data_target)
    
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据集。
        @params:
            index: int 索引
        @return:
            x: torch.Tensor 数据
            y: torch.Tensor 标签
        """
        data_idx = self.data_target[index][0]
        data = np.load(os.path.join(self.processed_folder, "%d.npy" % (data_idx,)))
        data = self.tpyx_2_tensor(data)
        target = self.data_target[index][1]
        return data, target


class CIFAR10DVS(AEDAT):
    original_size = (0, 2, 128, 128)
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
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    y_mask = 0x7F00
    y_shift = 8
    x_mask = 0x00FE
    x_shift = 1
    p_mask = 0x0001
    p_shift = 0


    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, time_steps: int = 64, width: int = 128, height: int = 128, polarity: bool = True, clipped: Optional[Union[Tuple, int]] = None) -> None:
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
            clipped: bool 要在t为什么范围内截取事件，接受None（不截取）、int（结尾）或tuple（开头与结尾）
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
        if isinstance(clipped, Tuple):
            assert clipped[1] > clipped[0], "Clip end must be larger than clip start."
        self.clipped = clipped


    def check_exists(self) -> bool:
        """
        检查是否存在
        @return:
            if_exist: bool 是否存在
        """
        return all(
            check_integrity(os.path.join(self.raw_folder, filename)) for fileurl, filename, md5 in self.resources
        )


    def download(self) -> None:
        """
        下载CIFAR-10 DVS数据集
        """
        if self.check_exists():
            return
        os.makedirs(self.raw_folder, exist_ok = True)
        for fileurl, filename, md5 in self.resources:
            if os.path.isfile(os.path.join(self.raw_folder, filename)):
                print("[blue]File %s has already existed.[/blue]" % (os.path.join(self.raw_folder, filename),))
                continue
            is_downloaded = False
            for mirror in self.mirrors:
                url = mirror + fileurl
                try:
                    print("[blue]Downloading %s from %s.[/blue]" % (os.path.join(self.raw_folder, filename), url))
                    download_url(url, root = self.raw_folder, filename = filename, md5 = md5)
                    is_downloaded = True
                    break
                except URLError as error:
                    print("[red]Error in file %s downloaded from %s:\r\n\r\n    %s\r\n\r\nPlease manually download it.[/red]" % (os.path.join(self.raw_folder , filename), url, error))
                    is_downloaded = False
            if is_downloaded:
                print("[green]Successfully downloaded %s.[/green]" % (os.path.join(self.raw_folder, filename),))
    

    def unzip(self) -> None:
        zip_file_list = os.listdir(self.raw_folder)
        extracted_folder_list = os.listdir(self.extracted_folder)
        if len(zip_file_list) == len(extracted_folder_list):
            print("[blue]Files are already extracted.[/blue]")
            return
        error_occured = False
        for filename in zip_file_list:
            label = self.label(filename.split(".")[0])
            if not os.path.isdir(self.extracted_folder):
                os.makedirs(self.extracted_folder, exist_ok = True)
            try:
                extract_archive(os.path.join(self.raw_folder, filename), self.extracted_folder)
                print("[green]Sussessfully extracted file %s.[/green]" % (filename,))
            except BadZipFile as e:
                print("[red]Error in unzipping file %s:\r\n\r\n    %s\r\n\r\nPlease manually fix the problem.[/red]" % (filename, e))
                error_occured = True
        if error_occured:
            raise RuntimeError("There are error(s) in unzipping files.")
        
    
    def load_data(self) -> np.ndarray:
        """
        加载数据集。
        @return:
            data_label: np.ndarray 数据信息，包括3列：数据集、标签、其为训练集（1）还是测试集（0）。
        """
        list_filename = os.path.join(self.processed_folder, "__main__.csv")
        if os.path.isfile(list_filename):
            file_list = np.loadtxt(list_filename, dtype = np.int, delimiter = ",")
            return file_list
        self.unzip()
        os.makedirs(self.processed_folder, exist_ok = True)
        file_list = []
        file_idx = 0
        for label_str in self.labels:
            label = self.label(label_str)
            aedat_files = os.listdir(os.path.join(self.extracted_folder, label_str))
            aedat_file_count = len(aedat_files)
            for filename in track(aedat_files, description = "Processing label %s" % (label_str,)):
                raw_data = self.filename_2_data(os.path.join(self.extracted_folder, label_str, filename))
                event_data = self.data_2_tpyx(raw_data)
                np.save(os.path.join(self.processed_folder, "%d.npy" % (file_idx,)), event_data)
                file_list.append([file_idx, label, 1 if self.is_train(label, file_idx % aedat_file_count) else 0])
                file_idx += 1
        file_list = np.array(file_list, dtype = np.int)
        np.savetxt(list_filename, file_list, fmt = "%d", delimiter = ",")
        return file_list
    

    def tpyx_2_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        将t,p,y,x数组转为最后的PyTorch张量。
        @params:
            data: np.ndarray 数据，形状为[n, 4]
        @return:
            data_tensor: torch.Tensor 渲染成事件的张量，形状为[T, C(P), H, W]
        """
        res = torch.zeros(self.t_size, self.p_size, self.y_size, self.x_size)
        if self.clipped is not None:
            if isinstance(self.clipped, int):
                data = data[data[:, 0] < self.clipped]
            elif isinstance(self.clipped, Tuple):
                data = data[(data[:, 0] >= self.clipped[0]) and (data[:, 0] < self.clipped[1])]
        data[:, 0] = np.floor(data[:, 0] * self.t_size / max(np.max(data[:, 0]) + 1, 1))
        data[:, 1] = np.floor(data[:, 1] * self.p_size / self.original_size[1])
        data[:, 2] = np.floor(data[:, 2] * self.y_size / self.original_size[2])
        data[:, 3] = np.floor(data[:, 3] * self.x_size / self.original_size[3])
        data = np.unique(data, axis = 0)
        res[data.T] = 1
        return res