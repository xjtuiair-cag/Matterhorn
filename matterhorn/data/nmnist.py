from zipfile import BadZipFile
import numpy as np
import torch
import torch.nn as nn
import os
import re
import shutil
import random
from torchvision.datasets.utils import  check_integrity, download_url, extract_archive, verify_str_arg
from torch.utils.data import Dataset, DataLoader
from typing import Any, List, Tuple, Union, Callable, Optional
from urllib.error import URLError
from rich import print
from rich.progress import track


class NMNIST(Dataset):
    training_file = "training.pt"
    test_file = "test.pt"
    original_size = (0, 2, 34, 34)
    mirrors = ["https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"]
    resources = [
        ("39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded", "Train.zip"),
        ("05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded", "Test.zip")
    ]
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, time_steps: int = 64, width: int = 34, height: int = 34, polarity: bool = True, clipped: Optional[Union[Tuple, int]] = None) -> None:
        """
        NMNIST数据集，将MNIST数据集动态变换后，转为事件的形式。
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
            clipped: bool 要在t为什么范围内截取事件，接受None（不截取）、int（结尾）或tuple（开头与结尾）
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.t_size = time_steps
        self.p_size = 2 if polarity else 1
        self.x_size = width
        self.y_size = height
        if isinstance(clipped, Tuple):
            assert clipped[1] > clipped[0], "Clip end must be larger than clip start."
        self.clipped = clipped
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
        return all(
            check_integrity(os.path.join(self.raw_folder, filename)) for fileurl, filename in self.resources
        )
        
    
    def download(self) -> None:
        """
        下载数据集。
        """
        if self.check_exists():
            return
        os.makedirs(self.raw_folder, exist_ok = True)
        for fileurl, filename in self.resources:
            if os.path.isfile(os.path.join(self.raw_folder, filename)):
                print("[blue]File %s has already existed.[/blue]" % (os.path.join(self.raw_folder, filename),))
                continue
            is_downloaded = False
            for mirror in self.mirrors:
                url = mirror + fileurl
                try:
                    print("[blue]Downloading %s from %s.[/blue]" % (os.path.join(self.raw_folder, filename), url))
                    download_url(url, root = self.raw_folder, filename = filename)
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
        data = np.fromstring(data_str, dtype = ">u1")
        return data


    def data_2_tpyx(self, data: np.ndarray) -> np.ndarray:
        """
        将数据分割为t,p,y,x数组。
        @params:
            data: np.ndarray 数据，形状为[2n]
        @return:
            data_tpyx: np.ndarray 分为t,p,y,x的数据，形状为[n, 4]
        """
        res = np.zeros((data.shape[0] // 5, 4), dtype = np.int)
        res[:, 0] = ((data[2::5] & 0x7f) << 16) + (data[3::5] << 8) + data[4::5]
        res[:, 1] = data[2::5] >> 7
        res[:, 2] = data[1::5]
        res[:, 3] = data[::5]
        return res

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
            is_train_str = "Train" if is_train else "Test"
            for label in track(range(len(self.labels)), description = "Processing %sing set" % (is_train_str.lower(),)):
                label_str = self.labels[label]
                raw_file_dir = os.path.join(self.extracted_folder, is_train_str, label_str)
                raw_file_list = os.listdir(raw_file_dir)
                for raw_filename in raw_file_list:
                    raw_data = self.filename_2_data(os.path.join(raw_file_dir, raw_filename))
                    event_data = self.data_2_tpyx(raw_data)
                    np.save(os.path.join(self.processed_folder, "%d.npy" % (file_idx,)), event_data)
                    file_list.append([file_idx, label, is_train])
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
                data = data[(data[:, 0] >= self.clipped[0]) & (data[:, 0] < self.clipped[1])]
        data[:, 0] = np.floor(data[:, 0] * self.t_size / max(np.max(data[:, 0]) + 1, 1))
        data[:, 1] = np.floor(data[:, 1] * self.p_size / self.original_size[1])
        data[:, 2] = np.floor(data[:, 2] * self.y_size / self.original_size[2])
        data[:, 3] = np.floor(data[:, 3] * self.x_size / self.original_size[3])
        data = np.unique(data, axis = 0)
        res[data.T] = 1
        return res


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