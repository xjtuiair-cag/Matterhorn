# -*- coding: UTF-8 -*-
"""
数据集的基本框架，定义了常见的数据集参数和处理方式。参考torchvision的MNIST数据集。
"""


import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity
from typing import Tuple, Iterable, Union, Callable, Optional
from rich import print
from rich.progress import track
import os
import json
import time
import shutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import matterhorn_pytorch.data.functional as DF


class EventDataset(Dataset):
    idx_filename = "__main__.csv"
    original_data_polarity_exists = False
    original_size = (1,)
    mirrors = []
    resources = []
    labels = []
    shape = None
    data_target = []


    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, cached: bool = True, cache_dtype: torch.dtype = torch.uint8, count: bool = False) -> None:
        """
        事件数据集框架
        Args:
            root (str): 数据集的存储位置
            train (bool): 是否为训练集
            transform (Callable | None): 数据如何变换
            target_transform (Callable | None): 标签如何变换
            download (bool): 如果数据集不存在，是否应该下载
            cached (bool): 是否为数据集作缓存。若为 False，则不作缓存，但是代价是运行速度变慢
            cache_dtype (torch.dtype): 如果为数据集作缓存，缓存的数据类型。默认为8位整型数，若count=True，您可能需要更高的精度储存脉冲张量
            count (bool): 是否采取脉冲计数，若为True则输出张量中各个点脉冲的个数，否则只输出是否有脉冲
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.cached = cached
        self.cache_dtype = cache_dtype
        self.count = count
        if download:
            self.download()
        self.pre_process()


    @property
    def raw_folder(self) -> str:
        """
        刚下载下来的数据集所存储的地方。
        Returns:
            res (str): 数据集存储位置
        """
        return os.path.join(self.root, self.__class__.__name__, "raw")


    @property
    def extracted_folder(self) -> str:
        """
        解压过后的数据集所存储的地方。
        Returns:
            res (str): 数据集存储位置
        """
        return os.path.join(self.root, self.__class__.__name__, "extracted")


    @property
    def processed_folder(self) -> str:
        """
        处理过后的数据集所存储的地方。
        Returns:
            res (str): 数据集存储位置
        """
        return os.path.join(self.root, self.__class__.__name__, "processed")


    @property
    def cached_folder(self) -> str:
        """
        张量缓存所存储的地方。
        Returns:
            res (str): 张量缓存位置
        """
        return os.path.join(self.root, self.__class__.__name__, "cached")


    @property
    def cache_tag(self) -> str:
        """
        张量缓存标签。
        Returns:
            res (str): 张量缓存标签
        """
        appendix = "_%s" % (str(self.cache_dtype))
        if self.transform is not None:
            appendix += "_" + hashlib.md5(repr(self.transform).encode("utf-8")).hexdigest()
        if self.count:
            appendix += "_count"
        return "x".join([str(i) for i in self.shape]) + appendix


    def check_exists(self) -> bool:
        """
        检查是否存在。
        Returns:
            if_exist (bool): 是否存在
        """
        return all(
            check_integrity(os.path.join(self.raw_folder, filename)) for fileurl, filename, md5 in self.resources
        )


    @staticmethod
    def clip_bits(data: np.ndarray, mask: int, shift: int) -> np.ndarray:
        """
        从事件数据中提取x,y,p值所用的函数。
        Args:
            data (np.ndarray): 事件数据
            mask (int): 对应的掩模
            shift (int): 对应的偏移量
        Returns:
            data (np.ndarray): 处理后的数据（x,y,p）
        """
        return (data >> shift) & mask


    def create_processed(self, data_idx: int, event_data: np.ndarray) -> bool:
        """
        将预处理的数据存储至文件中。
        Args:
            data_idx (int): 数据序号
            event_data (np.ndarray): 事件数据
        """
        np.save(os.path.join(self.processed_folder, "%d.npy" % (data_idx,)), event_data)
        return True


    def clear_processed(self, except_index: bool = False) -> None:
        """
        清空processed文件夹中的内容。
        Args:
            except_index (bool): 是否要排除"__main__.csv"
        """
        if os.path.isdir(self.processed_folder):
            shutil.rmtree(self.processed_folder)
        self.clear_cache()


    def create_cache(self) -> None:
        """
        创建缓存。
        """
        info_json = os.path.join(self.cached_folder, "__info__.json")
        cache_info = {
            "type": self.cache_tag,
            "object_num": len(self.data_target),
            "last_modified": time.time()
        }

        # 打开缓存文件夹，查看基本信息
        if os.path.isdir(self.cached_folder) and os.path.isfile(info_json):   
            with open(info_json, "r", encoding = "utf-8") as f:
                old_cache_info = json.load(f)
            cache_type = old_cache_info["type"]
            cache_size = old_cache_info["object_num"]
            cache_mtime = old_cache_info["last_modified"]
            file_num = sum([1 if s.endswith(".pt") else 0 for s in os.listdir(self.cached_folder)])

            # 如果类型对得上，且文件个数一致，就更新基本信息，使用原缓存
            if cache_type == self.cache_tag and cache_size == len(self.data_target) and cache_size == file_num:
                print("[green]Using cache file.[/green]")
                with open(info_json, "w", encoding = "utf-8") as f:
                    json.dump(cache_info, f)
                return
        
        # 若打不开缓存文件夹或缓存文件夹不满足条件，清除原有的其它缓存，重建新的缓存
        self.clear_cache()
        os.makedirs(self.cached_folder, exist_ok = True)
        print("[blue]Making cache, please wait ...[/blue]")

        def create_cache_file(source: str, dest: str, convert: Callable, dtype: torch.dtype) -> None:
            """
            将数据从源地址加载出来后，经过转换，放入目标地址。
            Args:
                source (str): 源地址
                dest (str): 目标地址
                convert (Callable): 转换规则
            """
            if os.path.isfile(dest):
                return
            data = np.load(source)
            data = convert(data)
            data = data.to(dtype)
            torch.save(data, dest)
        
        # 使用多线程进行转换。为保险起见，用最多2*CPU个worker，每次处理4*CPU个数据
        workers_num = multiprocessing.cpu_count() * 2
        batch_size = workers_num * 4
        looping = ((len(self.data_target) - 1) // batch_size) + 1

        # 循环处理每个数据，得到缓存
        for i in track(range(looping), description = "Making cache"):
            with ThreadPoolExecutor(max_workers = workers_num) as t:
                # 执行线程
                task_pool = []
                for j in range(batch_size):
                    idx = i * batch_size + j
                    if idx >= len(self.data_target):
                        break
                    data_idx = self.data_target[idx][0]
                    source_dir = os.path.join(self.processed_folder, "%d.npy" % (data_idx,))
                    target_dir = os.path.join(self.cached_folder, "%d.pt" % (data_idx,))
                    task_pool.append(t.submit(create_cache_file, source_dir, target_dir, self.event_data_to_tensor, self.cache_dtype))
                wait(task_pool)

                # 检查线程有无出错
                for k, t in enumerate(task_pool):
                    t = task_pool[k]
                    if t.exception():
                        print("[red bold]Error occured in thread %d:[/red bold]" % (idx,))
                        raise RuntimeError(t.exception())
        
        # 写入基本信息
        print("[green]Successfully made cache of %d data.[/green]" % (len(self.data_target,)))
        with open(info_json, "w", encoding = "utf-8") as f:
            json.dump(cache_info, f)


    def clear_cache(self) -> None:
        """
        清除缓存。
        """
        if os.path.isdir(self.cached_folder):
            shutil.rmtree(self.cached_folder)


    def download(self) -> None:
        """
        下载数据集。
        """
        return


    def process(self) -> np.ndarray:
        """
        加载数据集。
        Returns:
            data_label (np.ndarray): 数据信息，包括3列：数据集、标签、其为训练集（1）还是测试集（0）。
        """
        return None


    def pre_process(self) -> None:
        """
        数据的预处理，可以自定义。
        """
        if not self.check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        self.data_target = self.process()
        if self.cached:
            self.create_cache()
        self.data_target = self.data_target[self.data_target[:, 2] == (1 if self.train else 0)][:, :2]
        self.data_target = self.data_target.tolist()


    def event_data_to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        将缓存的numpy矩阵转为最后的PyTorch张量。
        Args:
            data (np.ndarray): 数据
        Returns:
            data_tensor (torch.Tensor): 渲染成事件的张量
        """
        if self.transform is not None:
            data = self.transform(data)
        return data


    def __len__(self) -> int:
        """
        获取数据集长度。
        Returns:
            len (int): 数据集长度
        """
        return len(self.data_target)
    
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据集。
        Args:
            index (int): 索引
        Returns:
            x (torch.Tensor): 数据
            y (torch.Tensor): 标签
        """
        data_idx = self.data_target[index][0]
        if self.cached:
            data = torch.load(os.path.join(self.cached_folder, "%d.pt" % (data_idx,)))
            data = data.to(torch.float)
        else:
            source_dir = os.path.join(self.processed_folder, "%d.npy" % (data_idx,))
            data = np.load(source_dir)
            data = self.event_data_to_tensor(data)
        target = self.data_target[index][1]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target


class EventDataset1d(EventDataset):
    original_size = (1, 2, 128)


    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, cached: bool = True, cache_dtype: torch.dtype = torch.uint8, count: bool = False, t_size: int = 128, x_size: int = 128, polarity: bool = True) -> None:
        """
        一维事件数据集框架
        Args:
            root (str): 数据集的存储位置
            train (bool): 是否为训练集
            transform (Callable | None): 数据如何变换
            target_transform (Callable | None): 标签如何变换
            download (bool): 如果数据集不存在，是否应该下载
            cached (bool): 是否为数据集作缓存。若为 False，则不作缓存，但是代价是运行速度变慢
            cache_dtype (torch.dtype): 如果为数据集作缓存，缓存的数据类型。默认为8位整型数，若count=True，您可能需要更高的精度储存脉冲张量
            count (bool): 是否采取脉冲计数，若为True则输出张量中各个点脉冲的个数，否则只输出是否有脉冲
            t_size (int): 时间维度的大小
            x_size (int): 空间维度的大小
            polarity (bool): 最终数据集是否采集极性信息，如果采集，通道数就是2，否则是1
        """
        self.t_size = t_size
        self.p_size = 2 if polarity else 1
        self.x_size = x_size
        self.shape = (self.t_size, self.p_size, self.x_size)
        super().__init__(
            root = root,
            train = train,
            transform = transform,
            target_transform = target_transform,
            download = download,
            cached = cached,
            cache_dtype = cache_dtype,
            count = count
        )
    

    def tx_to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        将t,x数组转为最后的PyTorch张量。
        Args:
            data (np.ndarray): 数据，形状为[n, 2]
        Returns:
            data_tensor (torch.Tensor): 渲染成事件的张量，形状为[T, L]
        """
        res = DF.event_seq_to_spike_train(
            event_seq = data,
            shape = (self.t_size, self.x_size),
            original_shape = (max(np.max(data[:, 0]) + 1, self.original_size[0]), self.original_size[2]),
            count = self.count
        )
        return res
    

    def tpx_to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        将t,p,x数组转为最后的PyTorch张量。
        Args:
            data (np.ndarray): 数据，形状为[n, 3]
        Returns:
            data_tensor (torch.Tensor): 渲染成事件的张量，形状为[T, C(P), L]
        """
        res = DF.event_seq_to_spike_train(
            event_seq = data,
            shape = (self.t_size, self.p_size, self.x_size),
            original_shape = (max(np.max(data[:, 0]) + 1, self.original_size[0]), self.original_size[1], self.original_size[2]),
            count = self.count
        )
        return res


    def event_data_to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        将缓存的numpy矩阵转为最后的PyTorch张量。
        Args:
            data (np.ndarray): 数据
        Returns:
            data_tensor (torch.Tensor): 渲染成事件的张量
        """
        data = super().event_data_to_tensor(data)
        if self.original_data_polarity_exists:
            return self.tpx_to_tensor(data)
        return self.tx_to_tensor(data)


class EventDataset2d(EventDataset):
    original_size = (1, 2, 128, 128)


    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, cached: bool = True, cache_dtype: torch.dtype = torch.uint8, count: bool = False, t_size: int = 128, y_size: int = 128, x_size: int = 128, polarity: bool = True) -> None:
        """
        二维事件数据集框架
        Args:
            root (str): 数据集的存储位置
            train (bool): 是否为训练集
            transform (Callable | None): 数据如何变换
            target_transform (Callable | None): 标签如何变换
            download (bool): 如果数据集不存在，是否应该下载
            cached (bool): 是否为数据集作缓存。若为 False，则不作缓存，但是代价是运行速度变慢
            cache_dtype (torch.dtype): 如果为数据集作缓存，缓存的数据类型。默认为8位整型数，若count=True，您可能需要更高的精度储存脉冲张量
            count (bool): 是否采取脉冲计数，若为True则输出张量中各个点脉冲的个数，否则只输出是否有脉冲
            t_size (int): 时间维度的大小
            y_size (int): 第一个空间维度的大小
            x_size (int): 第二个空间维度的大小
            polarity (bool): 最终数据集是否采集极性信息，如果采集，通道数就是2，否则是1
        """
        self.t_size = t_size
        self.p_size = 2 if polarity else 1
        self.y_size = y_size
        self.x_size = x_size
        self.shape = (self.t_size, self.p_size, self.y_size, self.x_size)
        super().__init__(
            root = root,
            train = train,
            transform = transform,
            target_transform = target_transform,
            download = download,
            cached = cached,
            cache_dtype = cache_dtype,
            count = count
        )


    def tyx_to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        将t,y,x数组转为最后的PyTorch张量。
        Args:
            data (np.ndarray): 数据，形状为[n, 3]
        Returns:
            data_tensor (torch.Tensor): 渲染成事件的张量，形状为[T, H, W]
        """
        res = DF.event_seq_to_spike_train(
            event_seq = data,
            shape = (self.t_size, self.y_size, self.x_size),
            original_shape = (max(np.max(data[:, 0]) + 1, self.original_size[0]), self.original_size[2], self.original_size[3]),
            count = self.count
        )
        return res


    def tpyx_to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        将t,p,y,x数组转为最后的PyTorch张量。
        Args:
            data (np.ndarray): 数据，形状为[n, 4]
        Returns:
            data_tensor (torch.Tensor): 渲染成事件的张量，形状为[T, C(P), H, W]
        """
        res = DF.event_seq_to_spike_train(
            event_seq = data,
            shape = (self.t_size, self.p_size, self.y_size, self.x_size),
            original_shape = (max(np.max(data[:, 0]) + 1, self.original_size[0]), self.original_size[1], self.original_size[2], self.original_size[3]),
            count = self.count
        )
        return res


    def event_data_to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        将缓存的numpy矩阵转为最后的PyTorch张量。
        Args:
            data (np.ndarray): 数据
        Returns:
            data_tensor (torch.Tensor): 渲染成事件的张量
        """
        data = super().event_data_to_tensor(data)
        if self.original_data_polarity_exists:
            return self.tpyx_to_tensor(data)
        return self.tyx_to_tensor(data)