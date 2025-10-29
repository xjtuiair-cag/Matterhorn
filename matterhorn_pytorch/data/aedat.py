# -*- coding: UTF-8 -*-
"""
AEDAT类数据集（后缀名为.aedat）。
"""


import numpy as np
import torch
import os
import re
import shutil
from typing import Callable as _Callable, Optional as _Optional
from rich import print
from rich.progress import track
import matterhorn_pytorch.data.functional as _DF
from matterhorn_pytorch.data.skeleton import EventDataset2d as _EventDataset2d


class AEDAT(_EventDataset2d):
    original_data_polarity_exists = True
    original_size = (1, 2, 128, 128)
    mirrors = []
    resources = []
    labels = []
    y_mask = 0x000007FF
    y_shift = 16
    x_mask = 0x000007FF
    x_shift = 1
    p_mask = 0x00000001
    p_shift = 0
    
    
    def __init__(self, root: str, train: bool = True, transform: _Optional[_Callable] = None, target_transform: _Optional[_Callable] = None, download: bool = False, cached: bool = True, cache_dtype: torch.dtype = torch.uint8, count: bool = False, time_steps: int = 128, width: int = 128, height: int = 128, polarity: bool = True, endian: str = ">", datatype: str = "u4") -> None:
        """
        原始数据后缀名为.aedat的数据集
        Args:
            root (str): 数据集的存储位置
            train (bool): 是否为训练集
            transform (_Callable | None): 数据如何变换
            target_transform (_Callable | None): 标签如何变换
            download (bool): 如果数据集不存在，是否应该下载
            cached (bool): 是否为数据集作缓存。若为 False，则不作缓存，但是代价是运行速度变慢
            cache_dtype (torch.dtype): 如果为数据集作缓存，缓存的数据类型。默认为8位整型数，若count=True，您可能需要更高的精度储存脉冲张量
            count (bool): 是否采取脉冲计数，若为True则输出张量中各个点脉冲的个数，否则只输出是否有脉冲
            time_steps (int): 最终的数据集总共含有多少个时间步
            width (int): 最终数据集的宽度
            height (int): 最终数据集的高度
            polarity (bool): 最终数据集是否采集极性信息，如果采集，通道数就是2，否则是1
            endian (str): 大端还是小端，">"代表大端存储，"<"代表小端存储
            datatype (str): 数据类型，如u4表示uint32
        """
        self.endian = endian
        self.datatype = datatype
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
    def filename_to_data(filename: str, dtype: str) -> np.ndarray:
        """
        输入文件名，读取文件内容。
        Args:
            filename (str): 文件名
            dtype (str): 数据类型
        Returns:
            data (np.ndarray): 文件内容（数据）
        """
        data_str = ""
        with open(filename, 'rb') as f:
            data_str = f.read()
            lines = data_str.split(b'\n')
            for line, contents in enumerate(lines):
                if not contents.startswith(b'#'):
                    break
            lines = lines[line:]
            data_str = b'\n'.join(lines)
        data = np.fromstring(data_str, dtype = dtype)
        return data
    
    
    @staticmethod
    def data_to_tpyx(data: np.ndarray, p_shift: int, p_mask: int, y_shift: int, y_mask: int, x_shift: int, x_mask: int) -> np.ndarray:
        """
        将数据分割为t,p,y,x数组。
        Args:
            data (np.ndarray): 数据，形状为[2n]
            p_shift (int): 极性p的偏移量
            p_mask (int): 极性p的掩膜
            y_shift (int): 坐标y的偏移量
            y_mask (int): 坐标y的掩膜
            x_shift (int): 坐标x的偏移量
            x_mask (int): 坐标x的掩膜
        Returns:
            data_tpyx (np.ndarray): 分为t,p,y,x的数据，形状为[n, 4]
        """
        res = np.zeros((data.shape[0] // 2, 4), dtype = "uint32")
        xyp = data[::2]
        t = data[1::2]
        res[:, 0] = t
        res[:, 1] = AEDAT.clip_bits(xyp, p_mask, p_shift)
        res[:, 2] = AEDAT.clip_bits(xyp, y_mask, y_shift)
        res[:, 3] = AEDAT.clip_bits(xyp, x_mask, x_shift)
        res = res[np.argsort(res[:, 0])]
        return res
    
    
    def label(self, key: str) -> int:
        """
        返回该文件对应的标签。
        Args:
            key (str): 关键词
        Returns:
            label (int): 文件的标签
        """
        if key in self.labels:
            return self.labels.index(key)
        return -1
    
    
    def is_train(self, label: int, index: int = 0) -> bool:
        """
        从路径、文件名和索引判断是否是训练集。
        Args:
            label (int): 标签
            index (int): 文件的索引
        Returns:
            is_train (bool): 是否为训练集
        """
        return True


class CIFAR10DVS(AEDAT):
    original_data_polarity_exists = True
    original_size = (1, 2, 128, 128)
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
    y_mask = 0x007F
    y_shift = 8
    x_mask = 0x007F
    x_shift = 1
    p_mask = 0x0001
    p_shift = 0


    def __init__(self, root: str, train: bool = True, transform: _Optional[_Callable] = None, target_transform: _Optional[_Callable] = None, download: bool = False, cached: bool = True, cache_dtype: torch.dtype = torch.uint8, count: bool = False, time_steps: int = 128, width: int = 128, height: int = 128, polarity: bool = True) -> None:
        """
        CIFAR-10 DVS数据集，将CIFAR10数据集投影至LCD屏幕后，用事件相机录制的数据集
        Args:
            root (str): 数据集的存储位置
            train (bool): 是否为训练集
            transform (_Callable | None): 数据如何变换
            target_transform (_Callable | None): 标签如何变换
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
            time_steps = time_steps,
            width = width,
            height = height,
            polarity = polarity,
            endian = ">",
            datatype = "u4"
        )


    def is_train(self, label: int, index: int = 0) -> bool:
        """
        从路径、文件名和索引判断是否是训练集。
        Args:
            label (int): 标签
            index (int): 文件的索引
        Returns:
            is_train (bool): 是否为训练集
        """
        return index < 900


    def download(self) -> None:
        """
        下载CIFAR-10 DVS数据集
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
        for label_str in self.labels:
            label = self.label(label_str)
            aedat_files = os.listdir(os.path.join(self.extracted_folder, label_str))
            aedat_file_count = len(aedat_files)
            for filename in track(aedat_files, description = "Processing label %s" % (label_str,)):
                raw_data = AEDAT.filename_to_data(os.path.join(self.extracted_folder, label_str, filename), self.endian + self.datatype)
                event_data = AEDAT.data_to_tpyx(raw_data, self.p_shift, self.p_mask, self.y_shift, self.y_mask, self.x_shift, self.x_mask)
                self.create_processed(file_idx, event_data)
                file_list.append([file_idx, label, 1 if self.is_train(label, file_idx % aedat_file_count) else 0])
                file_idx += 1
        file_list = np.array(file_list, dtype = "uint32")
        np.savetxt(list_filename, file_list, fmt = "%d", delimiter = ",")
        return file_list


class DVS128Gesture(AEDAT):
    original_data_polarity_exists = True
    original_size = (1, 2, 128, 128)
    mirrors = ["https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794"]
    resources = [
        ("", "DvsGesture.tar.gz", "8a5c71fb11e24e5ca5b11866ca6c00a1"),
        ("", "gesture_mapping.csv", "109b2ae64a0e1f3ef535b18ad7367fd1")
    ]
    labels = ["hand_clapping", "right_hand_wave", "left_hand_wave", "right_hand_clockwise", "right_hand_counter_clockwise", "left_hand_clockwise", "left_hand_counter_clockwise", "forearm_roll", "drums", "guitar", "random_other_gestures"]
    y_mask = 0x1FFF
    y_shift = 2
    x_mask = 0x1FFF
    x_shift = 17
    p_mask = 0x0001
    p_shift = 1


    def __init__(self, root: str, train: bool = True, transform: _Optional[_Callable] = None, target_transform: _Optional[_Callable] = None, download: bool = False, cached: bool = True, cache_dtype: torch.dtype = torch.uint8, count: bool = False, time_steps: int = 128, width: int = 128, height: int = 128, polarity: bool = True) -> None:
        """
        DVS128 Gesture数据集，用事件相机录制手势形成的数据集
        Args:
            root (str): 数据集的存储位置
            train (bool): 是否为训练集
            transform (_Callable | None): 数据如何变换
            target_transform (_Callable | None): 标签如何变换
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
            time_steps = time_steps,
            width = width,
            height = height,
            polarity = polarity,
            endian = "<",
            datatype = "u4"
        )


    @staticmethod
    def skip_header(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        cursor = 0
        while cursor < len(data):
            header = data[cursor:cursor + 7]
            event_type = (header[0] >> 16) & 0xFFFF # 1
            event_source = header[0] & 0xFFFF # 1
            event_size = header[1] # 8
            event_ts_offset = header[2] # 4
            event_ts_overflow = header[3] # 0
            event_capacity = header[4] # n
            event_number = header[5] # n ?
            event_valid = header[6] # n ?
            cursor += 7

            if event_type:
                end = cursor + (event_number * event_size // 4)
                mask[cursor:end] = True
                if event_ts_overflow:
                    data[cursor + 1:end:2] += event_ts_overflow << 31
            cursor += event_capacity * event_size // 4
        data = data[mask]
        return data
    

    @staticmethod
    def filename_to_data(filename: str, dtype: str) -> np.ndarray:
        """
        输入文件名，读取文件内容。
        Args:
            filename (str): 文件名
            dtype (str): 数据类型
        Returns:
            data (np.ndarray): 文件内容（数据）
        """
        data_str = ""
        with open(filename, 'rb') as f:
            data_str = f.read()
            lines = data_str.split(b'\n')
            for line, contents in enumerate(lines):
                if not contents.startswith(b'#'):
                    break
            lines = lines[line:]
            data_str = b'\n'.join(lines)
            data = np.fromstring(data_str, dtype = dtype)
            data = DVS128Gesture.skip_header(data, np.zeros(data.shape, dtype = "bool"))
        return data


    def is_train(self, label: int, index: int = 0) -> bool:
        """
        从路径、文件名和索引判断是否是训练集。
        Args:
            label (int): 标签
            index (int): 文件的索引
        Returns:
            is_train (bool): 是否为训练集
        """
        return index < 24


    def download(self) -> None:
        """
        下载CIFAR-10 DVS数据集
        """
        if self.check_exists():
            return
        os.makedirs(self.raw_folder, exist_ok = True)
        for fileurl, filename, md5 in self.resources:
            pathname = os.path.join(self.raw_folder, filename)
            for mirror in self.mirrors:
                url = mirror + fileurl
                # 本来就下不了，试过好几次了，所以直接报错完事
                print("[red]Cannot download file %s from %s, please manually download it.[/red]" % (pathname, url))


    def extract(self) -> None:
        """
        解压下载下来的压缩包。
        """
        os.makedirs(self.extracted_folder, exist_ok = True)
        filename = "DvsGesture.tar.gz"
        pathname = os.path.join(self.raw_folder, filename)
        success = _DF.unzip_file(pathname, self.extracted_folder)
        if not success:
            shutil.rmtree(self.extracted_folder)
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
        aedat_file_dir = os.path.join(self.extracted_folder, "DvsGesture")
        aedat_files = os.listdir(aedat_file_dir)
        os.makedirs(self.processed_folder, exist_ok = True)
        file_list = []
        file_idx = 0
        for filename in track(aedat_files, description = "Processing"):
            if not filename.endswith(".aedat"):
                continue
            raw_data = DVS128Gesture.filename_to_data(os.path.join(aedat_file_dir, filename), self.endian + self.datatype)
            event_data = DVS128Gesture.data_to_tpyx(raw_data, self.p_shift, self.p_mask, self.y_shift, self.y_mask, self.x_shift, self.x_mask)
            event_data[:, 2] = self.original_size[2] - 1 - event_data[:, 2]
            event_data[:, 3] = self.original_size[3] - 1 - event_data[:, 3]
            class_info = np.loadtxt(os.path.join(aedat_file_dir, filename.replace(".aedat", "_labels.csv")), dtype = "uint32", delimiter = ",", skiprows = 1)
            for label, start, end in class_info:
                data = event_data[(event_data[:, 0] >= start) & (event_data[:, 0] < end)]
                data[:, 0] = data[:, 0] - start
                user_id = re.search(r"user([0-9]+)", filename)
                user_id = user_id.group(1)
                is_train = self.is_train(label, int(user_id))
                self.create_processed(file_idx, event_data)
                file_list.append([file_idx, label - 1, 1 if is_train else 0])
                file_idx += 1
        file_list = np.array(file_list, dtype = "uint32")
        np.savetxt(list_filename, file_list, fmt = "%d", delimiter = ",")
        return file_list