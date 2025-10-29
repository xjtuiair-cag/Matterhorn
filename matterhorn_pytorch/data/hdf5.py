# -*- coding: UTF-8 -*-
"""
HDF5类数据集（后缀名为.h5）。
"""


import numpy as np
import torch
import os
import re
import shutil
from typing import Callable as _Callable, Optional as _Optional
from rich.progress import track
import matterhorn_pytorch.data.functional as _DF
from matterhorn_pytorch.data.skeleton import EventDataset1d
import h5py


class HDF5(EventDataset1d):
    original_data_polarity_exists = True
    training_file = "training.pt"
    test_file = "test.pt"
    original_size = (1, 2, 128)
    mirrors = []
    resources = []
    labels = []


    def __init__(self, root: str, train: bool = True, transform: _Optional[_Callable] = None, target_transform: _Optional[_Callable] = None, download: bool = False, cached: bool = True, cache_dtype: torch.dtype = torch.uint8, count: bool = False, precision: int = 1e9, time_steps: int = 128, length: int = 128) -> None:
        """
        原始数据后缀名为.hdf5的数据集
        Args:
            root (str): 数据集的存储位置
            train (bool): 是否为训练集
            transform (Callable | None): 数据如何变换
            target_transform (Callable | None): 标签如何变换
            download (bool): 如果数据集不存在，是否应该下载
            cached (bool): 是否为数据集作缓存。若为 False，则不作缓存，但是代价是运行速度变慢
            cache_dtype (torch.dtype): 如果为数据集作缓存，缓存的数据类型。默认为8位整型数，若count=True，您可能需要更高的精度储存脉冲张量
            count (bool): 是否采取脉冲计数，若为True则输出张量中各个点脉冲的个数，否则只输出是否有脉冲
            precision (int): 最终数据集的时间精度
            time_steps (int): 最终的数据集总共含有多少个时间步
            length (int): 最终数据集的空间精度
        """
        self.precision = precision
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
            x_size = length,
            polarity = False
        )


    @staticmethod
    def filename_to_data(filename: str) -> h5py.File:
        """
        输入文件名，读取文件内容。
        Args:
            filename (str): 文件名
        Returns:
            data (np.ndarray): 文件内容（数据）
        """
        data = h5py.File(filename, "r")
        return data


class SpikingHeidelbergDigits(HDF5):
    original_data_polarity_exists = False
    original_size = (1, 1, 700)
    mirrors = ["https://zenkelab.org/datasets/"]
    resources = [
        ('shd_train.h5.zip', 'shd_train.h5.zip', 'f3252aeb598ac776c1b526422d90eecb'),
        ('shd_test.h5.zip', 'shd_test.h5.zip', '1503a5064faa34311c398fb0a1ed0a6f')
    ]
    labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
    
    
    def __init__(self, root: str, train: bool = True, transform: _Optional[_Callable] = None, target_transform: _Optional[_Callable] = None, download: bool = False, cached: bool = True, cache_dtype: torch.dtype = torch.uint8, count: bool = False, precision: float = 1e9, time_steps: int = 128, length: int = 700) -> None:
        """
        Spiking Heidelberg Digits数据集，记录下英文和德语的0-9（总共20类），并转换成长度为700的脉冲。
        Args:
            root (str): 数据集的存储位置
            train (bool): 是否为训练集
            transform (Callable | None): 数据如何变换
            target_transform (Callable | None): 标签如何变换
            download (bool): 如果数据集不存在，是否应该下载
            cached (bool): 是否为数据集作缓存。若为 False，则不作缓存，但是代价是运行速度变慢
            cache_dtype (torch.dtype): 如果为数据集作缓存，缓存的数据类型。默认为8位整型数，若count=True，您可能需要更高的精度储存脉冲张量
            count (bool): 是否采取脉冲计数，若为True则输出张量中各个点脉冲的个数，否则只输出是否有脉冲
            precision (float): 最终数据集的时间精度
            time_steps (int): 最终的数据集总共含有多少个时间步
            length (int): 最终数据集的空间精度
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
            precision = precision,
            time_steps = time_steps,
            length = length
        )
        
    
    def download(self) -> None:
        """
        下载数据集。
        """
        if self.check_exists():
            return
        os.makedirs(self.raw_folder, exist_ok = True)
        for fileurl, filename, md5 in self.resources:
            pathname = os.path.join(self.raw_folder, filename)
            for mirror in self.mirrors:
                url = mirror + fileurl
                if _DF.download_file(url, pathname, md5):
                    break


    def extract(self) -> None:
        """
        解压下载下来的压缩包。
        """
        zip_file_list = os.listdir(self.raw_folder)
        if not os.path.isdir(self.extracted_folder):
            os.makedirs(self.extracted_folder, exist_ok = True)
        success = True
        for filename in zip_file_list:
            pathname = os.path.join(self.raw_folder, filename)
            success = success and _DF.unzip_file(pathname, self.extracted_folder)
        if not success:
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
            is_train_str = "train" if is_train else "test"
            raw_data = SpikingHeidelbergDigits.filename_to_data(os.path.join(self.extracted_folder, "shd_%s.h5" % (is_train_str,)))
            label_list = raw_data["labels"][:]
            for idx, label in track(enumerate(label_list), description = "Processing %sing set" % (is_train_str,)):
                t = np.floor(raw_data["spikes"]["times"][idx] * self.precision).astype("uint32")
                x = raw_data["spikes"]["units"][idx]
                event_data = np.zeros((len(x), 2), dtype = "uint32")
                event_data[:, 0] = t
                event_data[:, 1] = x
                event_data = event_data[np.argsort(event_data[:, 0])]
                self.create_processed(file_idx, event_data)
                file_list.append([file_idx, label, is_train])
                file_idx += 1
        file_list = np.array(file_list, dtype = "uint32")
        np.savetxt(list_filename, file_list, fmt = "%d", delimiter = ",")
        return file_list