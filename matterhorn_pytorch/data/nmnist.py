# -*- coding: UTF-8 -*-
"""
NMNIST数据集。
"""


import numpy as np
import torch
import os
from torchvision.datasets.utils import check_integrity, download_url, extract_archive
from typing import Iterable, Union, Callable, Optional
from urllib.error import URLError
from zipfile import BadZipFile
from rich import print
from rich.progress import track
from matterhorn_pytorch.data.skeleton import EventDataset2d


class NMNIST(EventDataset2d):
    original_data_polarity_exists = True
    original_size = (1, 2, 34, 34)
    mirrors = ["https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"]
    resources = [
        ("39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded", "Train.zip", "20959b8e626244a1b502305a9e6e2031"),
        ("05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded", "Test.zip", "69ca8762b2fe404d9b9bad1103e97832")
    ]
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, cached: bool = True, cache_dtype: torch.dtype = torch.uint8, count: bool = False, time_steps: int = 128, width: int = 34, height: int = 34, polarity: bool = True) -> None:
        """
        NMNIST数据集，将MNIST数据集动态变换后，转为事件的形式。
        Args:
            root (str): 数据集的存储位置
            train (bool): 是否为训练集
            transform (Callable | None): 数据如何变换
            target_transform (Callable | None): 标签如何变换
            download (bool): 如果数据集不存在，是否应该下载
            cached (bool): 是否为数据集作缓存。若为 False，则不作缓存，但是代价是运行速度变慢
            cache_dtype (torch.dtype): 如果为数据集作缓存，缓存的数据类型。默认为8位整型数，若count=True，您可能需要更高的精度储存脉冲张量
            count (bool): 是否采取脉冲计数，若为True则输出张量中各个点脉冲的个数，否则只输出是否有脉冲
            time_steps (int): 最终的数据集总共含有多少个时间步
            width (int): 最终数据集的宽度
            height (int): 最终数据集的高度
            polarity (bool): 最终数据集是否采集极性信息，如果采集，通道数就是2，否则是1
        """
        super().__init__(
            root = root,
            train = train,
            transform = transform,
            target_transform = target_transform,
            download = download,
            cached = cached,
            cache_dtype = cache_dtype,
            count = count,
            t_size = time_steps,
            y_size = height,
            x_size = width,
            polarity = polarity
        )


    @staticmethod
    def filename_to_data(filename: str) -> np.ndarray:
        """
        输入文件名，读取文件内容。
        Args:
            filename (str): 文件名
        Returns:
            data (np.ndarray): 文件内容（数据）
        """
        data_str = ""
        with open(filename, 'rb') as f:
            data_str = f.read()
        data = np.fromstring(data_str, dtype = ">u1")
        return data


    @staticmethod
    def data_to_tpyx(data: np.ndarray) -> np.ndarray:
        """
        将数据分割为t,p,y,x数组。
        Args:
            data (np.ndarray): 数据，形状为[5n]
        Returns:
            data_tpyx (np.ndarray): 分为t,p,y,x的数据，形状为[n, 4]
        """
        res = np.zeros((data.shape[0] // 5, 4), dtype = "uint32")
        res[:, 0] = ((data[2::5] & 0x7f) << 16) + (data[3::5] << 8) + data[4::5]
        res[:, 1] = data[2::5] >> 7
        res[:, 2] = 33 - data[1::5]
        res[:, 3] = data[::5]
        res = res[np.argsort(res[:, 0])]
        return res


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


    def extract(self) -> None:
        """
        解压下载下来的压缩包。
        """
        zip_file_list = os.listdir(self.raw_folder)
        if not os.path.isdir(self.extracted_folder):
            os.makedirs(self.extracted_folder, exist_ok = True)
        extracted_folder_list = os.listdir(self.extracted_folder)
        if len(zip_file_list) == len(extracted_folder_list):
            print("[blue]Files are already unzipped.[/blue]")
            return
        error_occured = False
        for filename in zip_file_list:
            print("[blue]Trying to unzip file %s ...[/blue]" % (filename,))
            try:
                extract_archive(os.path.join(self.raw_folder, filename), self.extracted_folder)
                print("[green]Sussessfully unzipped file %s.[/green]" % (filename,))
            except BadZipFile as e:
                print("[red]Error in unzipping file %s:\r\n\r\n    %s\r\n\r\nPlease manually fix the problem.[/red]" % (filename, e))
                error_occured = True
        if error_occured:
            raise RuntimeError("There are error(s) in unzipping files.")


    def process(self) -> np.ndarray:
        """
        加载数据集。
        Returns:
            data_label (np.ndarray): 数据信息，包括3列：数据集、标签、其为训练集（1）还是测试集（0）。
        """
        list_filename = os.path.join(self.processed_folder, self.idx_filename)
        if os.path.isfile(list_filename):
            file_list = np.loadtxt(list_filename, dtype = "uint32", delimiter = ",")
            return file_list
        self.extract()
        self.clear_processed()
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
                    raw_data = NMNIST.filename_to_data(os.path.join(raw_file_dir, raw_filename))
                    event_data = NMNIST.data_to_tpyx(raw_data)
                    self.create_processed(file_idx, event_data)
                    file_list.append([file_idx, label, is_train])
                    file_idx += 1
        file_list = np.array(file_list, dtype = "uint32")
        np.savetxt(list_filename, file_list, fmt = "%d", delimiter = ",")
        return file_list