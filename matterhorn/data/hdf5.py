import numpy as np
import torch
import os
import random
import h5py
from torchvision.datasets.utils import check_integrity, download_url, extract_archive
from typing import Any, List, Tuple, Union, Callable, Optional
from urllib.error import URLError
from zipfile import BadZipFile
from rich import print
from rich.progress import track
from matterhorn.data.skeleton import EventDataset1d


class HDF5(EventDataset1d):
    original_data_polarity_exists = True
    training_file = "training.pt"
    test_file = "test.pt"
    original_size = (1, 2, 128)
    mirrors = []
    resources = []
    labels = []


    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, sampling: int = 1, precision: int = 1e9, time_steps: int = 128, length: int = 128, clipped: Optional[Union[Tuple, float]] = None) -> None:
        """
        原始数据后缀名为.hdf5的数据集
        @params:
            root: str 数据集的存储位置
            train: bool 是否为训练集
            transform: Callable | None 数据如何变换
            target_transform: Callable | None 标签如何变换
            download: bool 如果数据集不存在，是否应该下载
            sampling: bool 是否进行采样（每隔n个事件采样一次），1为不采样（保存每个事件）
            precision: int 最终数据集的时间精度
            time_steps: int 最终的数据集总共含有多少个时间步
            length: int 最终数据集的空间精度
        """
        self.precision = precision
        super().__init__(
            root = root,
            train = train,
            transform = transform,
            target_transform = target_transform,
            download = download,
            sampling = sampling,
            t_size = time_steps,
            x_size = length,
            polarity = False,
            clipped = clipped
        )


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


class SpikingHeidelbergDigits(HDF5):
    original_data_polarity_exists = False
    original_size = (1, 1, 700)
    mirrors = ["https://zenkelab.org/datasets/"]
    resources = [
        ('shd_train.h5.zip', 'shd_train.h5.zip', 'f3252aeb598ac776c1b526422d90eecb'),
        ('shd_test.h5.zip', 'shd_test.h5.zip', '1503a5064faa34311c398fb0a1ed0a6f')
    ]
    labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
    
    
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, sampling: int = 1, precision: int = 1e9, time_steps: int = 128, length: int = 700, clipped: Optional[Union[Tuple, float]] = None) -> None:
        """
        Spiking Heidelberg Digits数据集，记录下英文和德语的0-9（总共20类），并转换成长度为700的脉冲。
        @params:
            root: str 数据集的存储位置
            train: bool 是否为训练集
            transform: Callable | None 数据如何变换
            target_transform: Callable | None 标签如何变换
            download: bool 如果数据集不存在，是否应该下载
            sampling: bool 是否进行采样（每隔n个事件采样一次），1为不采样（保存每个事件）
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
            sampling = sampling,
            precision = precision,
            time_steps = time_steps,
            length = length,
            clipped = clipped
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
        """
        解压下载下来的压缩包。
        """
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
                if self.sampling > 1:
                    event_data = event_data[::self.sampling]
                label = label_list[idx]
                self.save_event_data(file_idx, event_data)
                file_list.append([file_idx, label, is_train])
                file_idx += 1
        file_list = np.array(file_list, dtype = np.int)
        np.savetxt(list_filename, file_list, fmt = "%d", delimiter = ",")
        return file_list