from zipfile import BadZipFile
import numpy as np
import torch
import torch.nn as nn
import os
import random
import h5py
from torchvision.datasets.utils import  check_integrity, download_url, extract_archive, verify_str_arg
from torch.utils.data import Dataset, DataLoader
from typing import Any, List, Tuple, Union, Callable, Optional
from urllib.error import URLError
from rich import print
from rich.progress import track


class HDF5(Dataset):
    training_file = "training.pt"
    test_file = "test.pt"
    original_size = (0, 2, 128, 128)
    mirrors = []
    resources = []
    labels = []


    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, precision: int = 1e9, time_steps: int = 128, length: int = 128) -> None:
        """
        原始数据后缀名为.aedat的数据集
        @params:
            root: str 数据集的存储位置
            train: bool 是否为训练集
            transform: Callable | None 数据如何变换
            target_transform: Callable | None 标签如何变换
            download: bool 如果数据集不存在，是否应该下载
            precision: int 最终数据集的时间精度
            time_steps: int 最终的数据集总共含有多少个时间步
            length: int 最终数据集的空间精度
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.precision = precision
        self.t_size = time_steps
        self.x_size = length
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


    def filename_2_data(self, filename: str) -> h5py.File:
        """
        输入文件名，读取文件内容。
        @params:
            filename: str 文件名
        @return:
            data: np.ndarray 文件内容（数据）
        """
        data = h5py.File(filename, "r")
        return data
    
    
    def data_2_tensor(self, data: np.ndarray) -> torch.Tensor:
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
        return True
    
    
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
        data = self.data_2_tensor(data)
        target = self.data_target[index][1]
        return data, target



class SpikingHeidelbergDigits(HDF5):
    original_size = (0, 700)
    mirrors = ["https://zenkelab.org/datasets/"]
    resources = [
        ('shd_train.h5.zip', 'shd_train.h5.zip', 'f3252aeb598ac776c1b526422d90eecb'),
        ('shd_test.h5.zip', 'shd_test.h5.zip', '1503a5064faa34311c398fb0a1ed0a6f')
    ]
    labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
    
    
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, precision: int = 1e9, time_steps: int = 128, length: int = 700, clipped: Optional[Union[Tuple, float]] = None) -> None:
        """
        NMNIST数据集，将MNIST数据集动态变换后，转为事件的形式。
        @params:
            root: str 数据集的存储位置
            train: bool 是否为训练集
            transform: Callable | None 数据如何变换
            target_transform: Callable | None 标签如何变换
            download: bool 如果数据集不存在，是否应该下载
            precision: int 最终数据集的时间精度
            time_steps: int 最终的数据集总共含有多少个时间步
            length: int 最终数据集的空间精度
            clipped: bool 要在t为什么范围内截取事件，接受None（不截取）、int（结尾）或tuple（开头与结尾）
        """
        super().__init__(
            root = root,
            train = train,
            transform = transform,
            target_transform = target_transform,
            download = download,
            precision = precision,
            time_steps = time_steps,
            length = length
        )
        if isinstance(clipped, Tuple):
            assert clipped[1] > clipped[0], "Clip end must be larger than clip start."
        self.clipped = clipped
    
    
    def check_exists(self) -> bool:
        """
        检查是否存在。
        @return:
            if_exist: bool 是否存在
        """
        return all(
            check_integrity(os.path.join(self.raw_folder, filename)) for fileurl, filename, md5 in self.resources
        )
        
    
    def download(self) -> None:
        """
        下载数据集。
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
        if not os.path.isdir(self.extracted_folder):
            os.makedirs(self.extracted_folder, exist_ok = True)
        extracted_folder_list = os.listdir(self.extracted_folder)
        if len(zip_file_list) == len(extracted_folder_list):
            print("[blue]Files are already extracted.[/blue]")
            return
        error_occured = False
        for filename in zip_file_list:
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
        for is_train in range(2):
            is_train_str = "train" if is_train else "test"
            raw_data = self.filename_2_data(os.path.join(self.extracted_folder, "shd_%s.h5" % (is_train_str,)))
            label_list = raw_data["labels"][:]
            for idx in track(range(len(label_list)), description = "Processing %sing set" % (is_train_str,)):
                t = np.floor(raw_data["spikes"]["times"][idx] * self.precision).astype(np.int)
                x = raw_data["spikes"]["units"][idx]
                event_data = np.zeros((len(x), 2), dtype = np.int)
                event_data[:, 0] = t
                event_data[:, 1] = x
                label = label_list[idx]
                np.save(os.path.join(self.processed_folder, "%d.npy" % (file_idx,)), event_data)
                file_list.append([file_idx, label, is_train])
                file_idx += 1
        file_list = np.array(file_list, dtype = np.int)
        np.savetxt(list_filename, file_list, fmt = "%d", delimiter = ",")
        return file_list


    def data_2_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        将t,p,y,x数组转为最后的PyTorch张量。
        @params:
            data: np.ndarray 数据，形状为[n, 4]
        @return:
            data_tensor: torch.Tensor 渲染成事件的张量，形状为[T, C(P), H, W]
        """
        res = torch.zeros(self.t_size, self.x_size)
        if self.clipped is not None:
            if isinstance(self.clipped, int):
                data = data[data[:, 0] < self.clipped]
            elif isinstance(self.clipped, Tuple):
                data = data[(data[:, 0] >= self.clipped[0]) & (data[:, 0] < self.clipped[1])]
        data[:, 0] = np.floor(data[:, 0] * self.t_size / max(np.max(data[:, 0]) + 1, 1))
        data[:, 1] = np.floor(data[:, 1] * self.x_size / self.original_size[1])
        data = np.unique(data, axis = 0)
        res[data.T] = 1
        return res