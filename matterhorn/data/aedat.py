import numpy as np
import torch
import torch.nn as nn
import os
import re
import shutil
import random
from torchvision.datasets.utils import check_integrity, download_url, extract_archive
from typing import Any, List, Tuple, Union, Callable, Optional
from urllib.error import URLError
from zipfile import BadZipFile
from rich import print
from rich.progress import track
from matterhorn.data.skeleton import EventDataset2d


class AEDAT(EventDataset2d):
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
    
    
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, time_steps: int = 128, width: int = 128, height: int = 128, polarity: bool = True, endian: str = ">", datatype: str = "u4", clipped: Optional[Union[Tuple, int]] = None) -> None:
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
        self.endian = endian
        self.datatype = datatype
        super().__init__(
            root = root,
            train = train,
            transform = transform,
            target_transform = target_transform,
            download = download,
            t_size = time_steps,
            y_size = height,
            x_size = width,
            polarity = polarity,
            clipped = clipped 
        )


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


    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, time_steps: int = 128, width: int = 128, height: int = 128, polarity: bool = True, clipped: Optional[Union[Tuple, int]] = None) -> None:
        """
        CIFAR-10 DVS数据集，将CIFAR10数据集投影至LCD屏幕后，用事件相机录制的数据集
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
                self.save_event_data(file_idx, event_data)
                file_list.append([file_idx, label, 1 if self.is_train(label, file_idx % aedat_file_count) else 0])
                file_idx += 1
        file_list = np.array(file_list, dtype = np.int)
        np.savetxt(list_filename, file_list, fmt = "%d", delimiter = ",")
        return file_list


class DVS128Gesture(AEDAT):
    original_data_polarity_exists = True
    original_size = (1, 2, 128, 128)
    mirrors = ["https://public.boxcloud.com/d/1/"]
    resources = [
        ("b1!6RC4cE5cm0PrbRhQeTXuS96wG5z7ScLzlCbaTBHMZ0nP1ec3P-NwYIaFhlO_e275BKS0h_vaxU_a1VZ-vMNxfrGMvCuyD9Qb2L5qjyALlIBrFCODa_bCjOnX1IC47noyVjkoFK_i0F5eEDjjfjo58pSn-4-nzJ1eGxjbVvxRxxAVQzEhOqi3JezM2vkP_VT7gMUmObnhWOS02SgbXH8Aet4a1Lup4FMEA7RaQMUPFEOTRmcF54srarRQm7srTDWmu5GgE4mjRyBGcudExqoTKpgWfZBpzGA_c1j9f9h8HhquVQmcyiAtz4XBpI8Jsnca3Q2x5yuxreRDYxW9yrXLKikTOiUtDRxp0R75yk88Jtv_Af74Sf7hZI5U5GzpgJEsi6GObfXV_QTtWPurQ0O7Pr9wgMJSq6OvRn09Ei2PoNjgWnyj1nVR_2FL_s2PBOKMvx1A1d1KGF_BYdBhJcf1aewA09gNERRUHlw80TQoANFszTxOJ4wb8DuE4hYQ6k22m4l6SWTApMXtO-E4k8Apxwc5WqBZo_9pRdJc4W9XgTl3jTvgNvYAa95yifG4d_5vJnxbhdciVfsBBnHr-Xj3cyAS6NYL1wt_9A4z5KG0i1WizpRzTut1wwxA0t08qnMFZJHcsg2opVmvIKF1i9IG8TWBTHnbuZ25HjiZ3ogJX-dpBA9OPllpoEDLTMwVEjMvx3gGMiUgmF-TxKesMRSE8qnQx37mLXsc38rlBs6z-aPTR3P_1QTHIZTiLsZR4StnkuZB9RXaizj0d7PMN68QGi4_QEOW_c4bujAt_fXxbB1Grcvk57rl_MdfRFSH6qEYXHeYjngHFzBcomX0BVfwAjCNDWYp2kEiDe8YhxU3i44bDK3wG3yJG4Ia6dNKv7xzKWGPu3Wn-DSJB4Km0h9Vd7bCBQnV2Hp7kTPYuv8_UHcYtSOGgyqR_EkfWtejUAOG-FsKmLUs3Hx_YUePWIDgiYASSAzVFpr0T4qcWxl_Z-FksIpgjxDghyj6mepIrfHkEbY3Hi7HUgXOmaDQyO8tM3edHJIbyxtbHVbwdngy9jK-3uR4xzT5OSSzbTdOZFN_cwepwTZIslCrTfoxjGXnkMKE_4ldsGUszr5HbC2IRv1r1CQrHg7OORWJV4MyayfAyePNcPyo0x9w9BMCP13BZIS4SOxuHwYVjsCWYTpVE8Q4MbDT3ULIZ851vbeUSTWuAUSnhG98ByLC84vEAADuPbt3hnx_BnD6_fYtA7tWuiFqYZrbopIrTwKfo5L7FFj6usDf-mg3SD-V9PaMZM-qM-O6ZZLriUgpRfhkh78WlBU./download", "DvsGesture.tar.gz", "8a5c71fb11e24e5ca5b11866ca6c00a1"),
        ("b1!wW8v-TwIp6l_KNi3CTVIHiFXdJ2Zhio7A2setLDVT1CnhP4wO45mrTXeMgFGKcae8_BR078T5iHeVGSWyZDhzSp_5N6Jq8erTXgGL4CpP1t5i6dHcQIvbvDy-TGBI3q59Q3b03PEm98Q4v5dTMqTf1YvUijfirNTt-eakTrX0vfCKb4AJWmlme1LWrI_D5bfed7fqq73zcDsAzxrU3A0FV7MIwOZgXlUQan-2iDBqpqtFHTUE2eQLFWxJdF3pGPT7GlnzwK4glCakF6VSr30YnKje0Nom2aueDEn7WLuGQDcKLqssMYJSIT14l7lmHmYVGqwWxVxro-YSRKObd45_0sS3TUy3-m_uMGBvBVj7mJjCz51Rurg_b0QUeZOgoM3U3MAPHaU2Wuj1T5TgFjywwTlmJOZawjPVfg_TuwjROrWGe8b5b5cAkGvxFntK1MMeF_qNIOiQ7CuNdZYRkLNPVJwewgrjgVIle9xkH3_-BbbYMUfK5C5X7aFKjA7w6caI9nyQX8Zy3Q1Eac5giXBA6BGS6iJcuoCDZZcwQ5TWp7fIEafOTf1o-yHnr_bCWEuDIB5dzqpEleRAFgH7dXAZn6p5b9ZbtQPOaBkXeWP3ru_POVejZ0uccwYE17r_vd6ezSbf5DxKjCFXJMSxVsj7oH5Y9eGMqjhYZiRoHGuz1z_UQs4BZUOUyFDxFQS1zZtVW1gcbZbCJuAVEDMAG-ddx8Wk7zcZKzU5P6jWe1a1rjoUQiniRrGYq5h5q3QqMcX0KHtN4KqgEhrDvo8UGIlf75G-HX7dU8ktRQwVYApjbEvBk9D9fk7SHrxHkhXbSSvCixOE8vZFguKqW7dCLXBVEICSkJ4u1GusOrW4rDWqm06OE0gFdPe5LKVRptkCcCTpojhh_MdVeXoBcSlfJTT6jSutVZv4K8FJKu6K09Uv4u0CdOO--cORksy_zw3FC-BLUOPU8GzAKFuXCQH4xUo3zGQBvTbmf7V0qDsmcni2W6YKGtMQ9M25yIwIA3MPG1BWEA6M-sVWCMzPuHMq7OW4nl2eMLyIAeyirYprAJ2CJs9Vf8HycnTUYHuH5jZ4_-Ol0c5r3DFtEQrGZ7mc8XT9fDSgy306D9f52NVSO2okWU96PzSsDuk_Y0EunMverMo2EDCYjTOMgIAOsbd_cmRgDFgxX4w_nmf7ebFW5BreMCLCOyykyhWqRLwXifFDR10WUoDiAwq21vH_hYZLGqxL6UHBvuvOPRzepd2q-HipJfztOXlhVx_TScMAy-ZCyFiUPtJZMLK_V_y4a8W/download", "README.txt", "a0663d3b1d8307c329a43d949ee32d19"),
        ("b1!0t60iOKCh-vIhXawxHD_HGZO1UIWc9WrTYp2pjJHX01xQ-Hpih2ywvJTTXhLjqpSi3FITdH-dQtV-DhdWkx8MTMGcFcaz2bsrC3wKG6c4IQB_IxkCUc-GaEwqAMa1yz1nNZo1hB_LbyuDfSQcOy30KfWcna6RVk1Didf7RmtPDzc74oJQq5XgdAgJ4EM7hzJ-YcW7S-q707AfGeoPvp46N1F2A74GNIVR_JGyU_WC0AU4K44iHQH6ujSb6D-Bd6lZyaPi3pvBxSKf6S3k-ZuiNKHzPF4pICpNjQM-kbqneer9FljXLNv-zaiXPHtqcNVp6YTZw5TZmCdMFDsWkPVVQrsofl0Be_Hl2qX7vJM64KevJ9tHQHGWRzvpjGhBwYDh56XiwXkgsFC7KNtQoUQ4cRaYqtoJyUroNrSOOWXdvH7o-yLHC3RqL1LlIVLozGGAou4XNSRemR1JGQMJxuw2_XY8ovRLWotffqNVuSUNbk-6PQiugArV2VSFtAOQ6Hoo8gHtVmzrlGxQ3U1lfEZoKZmmxDHzMfRaLLplOnIJZDu-F2USrAaAOVOmJ580LuT4lZAoqQjx3dRidTo5J86Kd3cj90liFkVbAiko4pG2Jr0wMTgnCRBJufuawYkILgsE500-5ScnbcEbwUmPHqbySJl9gnT5haz83JfJPhkBK6GntUjy16ApLumIkhL7mABnPUkeDBNVeC73n5OUWdjb470xQLeWFqz6zCCnlEH31QS5PRe7Jfw2Dj4yst7ZkWpPUppIPKACMyGqboDAngOb7Kv0Cf37GuGpMkzs3Q54T4YKULXkc7wzoaVy1EUmt9zwci9n4u-tDDLDkhqj1wZ1_tFd2Y3Joydxte3bI3h64i5AKNNAcdAZ-7pkMT2fr-p5blRLGfEkXk2IObUbe22RX793JgxwNfD2laJhBBdrDLdKwNWtAIib4cdambhTHKovxno6YjQbyUgic4mCLRxojUVw02NjnF-UhH34wp8Dh-43_ibEWezqSLPp7PWutGlAgXD4PoDQeZumXHo0u7mGfO3GqkLr_1xlyBMVqfiW9Hy43DRWSRtlGmnNbgGN7gVlhcpGSdLuHPgzWGuI6-IkNXQPFmJwdMloizeNy9-ba2UASuOorvy4BaWiy36jDSsH9jJHbW1ffB5qsSchbUcKKlfgyx7SojhZq5xAw1sPD1QPer7xuu7ofPYJRmK_cQpMNWrUsGTuxWy8TOEUIiJNTYIIAznNoBDXaX6KqQggXt7stN26niyFgvWoKr1Cs7Z84Fy-xw./download", "LICENSE.txt", "065e10099753156f18f51941e6e44b66"),
        ("b1!0qorL_bz0q1mFjUK3_5eUQEG36l3-xxn2ahncEIaiPy0zuc238nf796Q7LUysNXYerX8LM_VPBAbqqwPxsAT6IdZtO3zAX2S4poHvangqpjMGw6Yez9dOD8rh--lhc5TSv7aQezqeMhOI-VS_Zf4PkxWMUXfmhzVY0kyo3rbhMvnVA5rAW4QciBazWLGQheSweM3iTOare3_P6OF830lGOge_1M26nzUTES56Sm19h_4sqb9_tgRzFOYRE5mk2IGUr5V3AOEm33CgZAuwh6RkKMqJeCXtwgyW3wz2t4oJE3suKisOWiPazmDLKpcgVhj2lssbMTeZKF6_MK6iNl_JabWxXi3X_4h-HxTH6iOzfXJd_Z76EU5YJWEyCKjc0sXp9-RirfcVua1dRtrnlPkd35rVOeOQvZzoc3i7Q9ftoR0sfJg3jA1inkTuuWULt5R4VBcJuVuIBzVT6dLI232yORqeA5MgoQCRjH_kGbQySEMWoFv8EP2xplhQRr6RkNuCx2_lMrPpiZYcrfbmxKhD_R9IsQ4XPRd-GNB_xBwfwuK9zrz4qnYHKqGjZf_F2jLpHLQMQVrFE7-LJZTRYQR2RkqWFHllJkKsnq6UmXvd7tAojv4p3LouiHdOar-yot3s8oNScLCcS_SM634d4sTyS-isGbzOKTw6l39rwzWOWnh0tVhyhpsqJJUVZvHzjYTs-VfD_RjLkFqGSkQvY9ChQxEC5mqUloRzq8fPYt__HqaBRXRFabvTp4jVIpFCNGYMtIaXJKsheovqtZ0OyqBFbsOKfXT9LLRuUuc73PXe0FO3u20WA3zyvWrdsobTQB078MUmOFAa15fJWjutIZ17XEvZovkNosI55cQ3RgRw_Fi05WVmT7ufWs-BPrQWvk7rnNNolChfHoW69DzjYTIPPK_aKOkljUSnf6MwRnZtf97xHxfsY-QWI0azcnMBkR6he2NdMaki5MryN3B1Yv3QsJSbyitASLxDUsN19ATPLnkrvxsXeF3bR7ATF4Rwcsf0fbiH_3EDnS1kE7PYUfeOUG6kV6QdlyqwTZOgjEicwpWYFZPvusUzWbifbhSrOY1fKuDO4aSfmum9TlXTMV2nFukKuEXhGKaFwj3Mpmva50VntJE5NHnpjzNfjCRC_vndQrCNRqvHJExIDsdVpBvOTsqctfwe5bGjBwaOEG5rwodQmPhcCx8HVGJho8scdByrorcz-9En8KjL-kTqnYkh0BY4tO9-GaQvtf3qAWS1I7EjZjVV8T3RmXn6TYocMGeuW6J04sMn5rLMGU./download", "gesture_mapping.csv", "109b2ae64a0e1f3ef535b18ad7367fd1")
    ]
    labels = ["hand_clapping", "right_hand_wave", "left_hand_wave", "right_hand_clockwise", "right_hand_counter_clockwise", "left_hand_clockwise", "left_hand_counter_clockwise", "forearm_roll_forward", "forearm_roll_backward", "drums", "guitar", "random_other_gestures"]
    y_mask = 0x1FFF
    y_shift = 2
    x_mask = 0x1FFF
    x_shift = 17
    p_mask = 0x0001
    p_shift = 1


    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, time_steps: int = 128, width: int = 128, height: int = 128, polarity: bool = True, clipped: Optional[Union[Tuple, int]] = None) -> None:
        """
        DVS128 Gesture数据集，用事件相机录制手势形成的数据集
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
            endian = "<",
            datatype = "u4"
        )
        if isinstance(clipped, Tuple):
            assert clipped[1] > clipped[0], "Clip end must be larger than clip start."
        self.clipped = clipped


    def download(self) -> None:
        """
        下载CIFAR-10 DVS数据集
        """
        if self.check_exists():
            return
        printed_download_url = "https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794"
        os.makedirs(self.raw_folder, exist_ok = True)
        for fileurl, filename, md5 in self.resources:
            if os.path.isfile(os.path.join(self.raw_folder, filename)):
                print("[blue]File %s has already existed.[/blue]" % (os.path.join(self.raw_folder, filename),))
                continue
            is_downloaded = False
            for mirror in self.mirrors:
                url = mirror + fileurl
                try:
                    print("[blue]Downloading %s from %s.[/blue]" % (os.path.join(self.raw_folder, filename), printed_download_url))
                    download_url(url, root = self.raw_folder, filename = filename, md5 = md5)
                    is_downloaded = True
                    break
                except URLError as error:
                    print("[red]Error in file %s downloaded from %s:\r\n\r\n    %s\r\n\r\nPlease manually download it.[/red]" % (os.path.join(self.raw_folder , filename), printed_download_url, error))
                    is_downloaded = False
            if is_downloaded:
                print("[green]Successfully downloaded %s.[/green]" % (os.path.join(self.raw_folder, filename),))


    def unzip(self) -> None:
        """
        解压下载下来的压缩包。
        """
        if os.path.isdir(self.extracted_folder):
            print("[blue]Files are already extracted.[/blue]")
            return
        os.makedirs(self.extracted_folder, exist_ok = True)
        filename = "DvsGesture.tar.gz"
        error_occured = False
        try:
            extract_archive(os.path.join(self.raw_folder, filename), self.extracted_folder)
            print("[green]Sussessfully extracted file %s.[/green]" % (filename,))
        except BadZipFile as e:
            print("[red]Error in unzipping file %s:\r\n\r\n    %s\r\n\r\nPlease manually fix the problem.[/red]" % (filename, e))
            error_occured = True
        if error_occured:
            shutil.rmtree(self.extracted_folder)
            raise RuntimeError("There are error(s) in unzipping files.")


    def is_train(self, label: int, index: int = 0) -> bool:
        """
        从路径、文件名和索引判断是否是训练集。
        @params:
            label: int 标签
            index: int 文件的索引
        @return:
            is_train: bool 是否为训练集
        """
        return index < 24


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
        aedat_file_dir = os.path.join(self.extracted_folder, "DvsGesture")
        aedat_files = os.listdir(aedat_file_dir)
        os.makedirs(self.processed_folder, exist_ok = True)
        file_list = []
        file_idx = 0
        for filename in track(aedat_files, description = "Processing"):
            if not filename.endswith(".aedat"):
                continue
            raw_data = self.filename_2_data(os.path.join(aedat_file_dir, filename))
            raw_data_list = []
            idx_list = np.where(raw_data == 65537)[0]
            for i in range(len(idx_list) - 1):
                raw_data_list.append(raw_data[idx_list[i] + 7: idx_list[i + 1]])
            raw_data_list.append(raw_data[idx_list[len(idx_list) - 1] + 7:])
            raw_data = np.concatenate(raw_data_list, axis = 0)
            event_data = self.data_2_tpyx(raw_data)
            class_info = np.loadtxt(os.path.join(aedat_file_dir, filename.replace(".aedat", "_labels.csv")), dtype = np.int, delimiter = ",", skiprows = 1)
            for label, start, end in class_info:
                data = event_data[(event_data[:, 0] >= start) & (event_data[:, 0] < end)]
                data[:, 0] = data[:, 0] - start
                user_id = re.search(r"user([0-9]+)", filename)
                user_id = user_id.group(1)
                is_train = self.is_train(label, int(user_id))
                self.save_event_data(file_idx, event_data)
                file_list.append([file_idx, label - 1, 1 if is_train else 0])
                file_idx += 1
        file_list = np.array(file_list, dtype = np.int)
        np.savetxt(list_filename, file_list, fmt = "%d", delimiter = ",")
        return file_list