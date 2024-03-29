import torch
from torch.utils.data import Dataset, DataLoader
import matterhorn_pytorch as mth
import time
import datetime
import os
from typing import Tuple, Dict, Union
from rich import print
from rich.panel import Panel
from rich.text import Text
from rich.progress import track
from rich.table import Table


def train_one_epoch(model: torch.nn.Module, data_loader: DataLoader, num_classes: int, optimizer: torch.optim.Optimizer = None, device: torch.device = None, dtype: torch.dtype = None) -> Tuple[torch.nn.Module, float, float]:
    """
    一个轮次的训练。
    Args:
        model (torch.nn.Module): 所使用的模型
        data_loader (DataLoader): 所使用的数据集
        num_classes (int): 最终类别个数
        optimizer (torch.optim.Optimizer): 所使用的参数优化器
        device (torch.device): 所使用的计算设备
        dtype (torch.dtype): 所使用的数据类型
    """
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_samples = 0

    for x, y in track(data_loader, description = "Training"):
        if optimizer is not None:
            optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y0 = torch.nn.functional.one_hot(y, num_classes = num_classes).float()
        o = model(x)
        loss = torch.nn.functional.mse_loss(o, y0)
        loss.backward()
        if optimizer is not None:
            optimizer.step()

        train_samples += y.numel()
        train_loss += loss.item() * y.numel()
        train_acc += (o.argmax(1) == y).float().sum().item()

    train_loss /= train_samples
    train_acc /= train_samples

    return model, train_loss, train_acc


def test_one_epoch(model: torch.nn.Module, data_loader: DataLoader, num_classes: int, device: torch.device = None, dtype: torch.dtype = None) -> Tuple[torch.nn.Module, float, float]:
    """
    一个轮次的测试。
    Args:
        model (torch.nn.Module): 所使用的模型
        data_loader (DataLoader): 所使用的数据集
        num_classes (int): 最终类别个数
        device (torch.device): 所使用的计算设备
        dtype (torch.dtype): 所使用的数据类型
    """
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_samples = 0

    with torch.no_grad():
        for x, y in track(data_loader, description = "Testing"):
            x = x.to(device)
            y = y.to(device)
            y0 = torch.nn.functional.one_hot(y, num_classes = num_classes).float()
            
            o = model(x)
            loss = torch.nn.functional.mse_loss(o, y0)

            test_samples += y.numel()
            test_loss += loss.item() * y.numel()
            test_acc += (o.argmax(1) == y).float().sum().item()
    
    test_loss /= test_samples
    test_acc /= test_samples

    return model, test_loss, test_acc


def init_logs(log_dir: str, model: torch.nn.Module) -> None:
    """
    初始化日志文件夹和日志文件。
    Args:
        log_dir (str): 存放日志的文件夹
        model (torch.nn.Module | torch.nn.Module*): 所使用的模型
    """
    os.makedirs(log_dir, exist_ok = True)
    with open(os.path.join(log_dir, "result.csv"), "w") as f:
        f.write("Epoch,Training Loss,Training Accuracy,Testing Loss,Testing Accuracy,Duration\n")
    save_model(log_dir, "last", model)


def print_title(title: str, style: str = "") -> None:
    """
    打印标题（带框）。
    Args:
        title (str): 标题内容
        style (str): 标题样式
    """
    print(Panel(Text(title, justify = "center", style = style)))


def print_params(param_dict: Dict) -> None:
    """
    打印超参数。
    Args:
        param_dict (str): 超参数列表
    """
    hyper_param_table = Table(show_header = True, header_style = "bold blue")
    hyper_param_table.add_column("Name", justify = "center")
    hyper_param_table.add_column("Value", justify = "center")
    for name in param_dict:
        hyper_param_table.add_row(name, str(param_dict[name]))
    print(hyper_param_table)


def save_model(path: str, name: str, model: torch.nn.Module):
    """
    存储模型。
    Args:
        path (str): 模型存储的路径
        name (str): 模型文件的命名
        model (torch.nn.Module | torch.nn.Module*): 所使用的模型
    """
    torch.save(model, os.path.join(path, "%s.pt" % (name)))


def print_model(model: torch.nn.Module) -> None:
    """
    打印模型。
    Args:
        model (torch.nn.Module | torch.nn.Module*): 所使用的模型
    """
    print(model)
    print()


def print_dataset(dataset: Dataset) -> None:
    """
    打印数据集简介。
    Args:
        dataset (Dataset): 所使用的数据集
    """
    demo_data, demo_label = dataset[0]
    info_table = Table(show_header = True, header_style = "bold blue")
    info_table.add_column("Dataset Name", justify = "center")
    info_table.add_column("Data Shape", justify = "center")
    info_table.add_row(dataset.__class__.__name__, str(demo_data.shape))
    print(info_table)


def print_result(ep: int, data: Tuple[float], last_data: Tuple[float], max_test_acc: float, duration: float) -> None:
    """
    打印结果。
    Args:
        ep (int): 当前轮次
        data (float*): 当前的数据，是一个元组，依次为(train_loss, train_acc, test_loss, test_acc)
        last_data (float*): 上一次的数据，是一个元组，依次为(last_train_loss, last_train_acc, last_test_loss, last_test_acc)
        max_test_acc (float): 最大测试准确率
        duration (float): 所经过的时间
    """
    train_loss, train_acc, test_loss, test_acc = data
    last_train_loss, last_train_acc, last_test_loss, last_test_acc = last_data
    get_color = lambda x, y: "green" if x > y else ("red" if x < y else "blue")

    result_table = Table(show_header = True, header_style = "bold blue")
    result_table.add_column("Name", justify = "center")
    result_table.add_column("Value", justify = "center")
    result_table.add_row("Epoch", str(ep + 1))
    result_table.add_row("Training Loss", "[%s]%g[/%s]" % (get_color(last_train_loss, train_loss), train_loss, get_color(last_train_loss, train_loss)))
    result_table.add_row("Training Accuracy", "[%s]%g%%[/%s]" % (get_color(train_acc, last_train_acc), 100 * train_acc,get_color(train_acc, last_train_acc)))
    result_table.add_row("Testing Loss", "[%s]%g[/%s]" % (get_color(last_test_loss, test_loss), test_loss, get_color(last_test_loss, test_loss)))
    result_table.add_row("Testing Accuracy", "[%s]%g%%[/%s]" % (get_color(test_acc, last_test_acc), 100 * test_acc, get_color(test_acc, last_test_acc)))
    result_table.add_row("Maximum Testing Accuracy", "%g%%" % (100 * max_test_acc,))
    result_table.add_row("Duration", "%gs" %(duration,))
    print(result_table)


def train_and_test(epochs: int, model: torch.nn.Module, train_data_loader: DataLoader, test_data_loader: DataLoader, num_classes: int, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None, log_dir: str = "./logs/", device: torch.device = None, dtype: torch.dtype = None) -> None:
    """
    训练模型。
    Args:
        epochs (int): 训练总轮次
        model (torch.nn.Module | torch.nn.Module*): 所使用的模型
        train_data_loader (DataLoader): 所使用的训练集
        test_data_loader (DataLoader): 所使用的测试集
        num_classes (int): 最终类别个数
        optimizer (torch.optim.Optimizer): 所使用的参数优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler): 所使用的学习率衰减器
        log_dir (str): 存放日志的文件夹
        device (torch.device): 所使用的计算设备
        dtype (torch.dtype): 所使用的数据类型
    """
    max_test_acc = 0.0
    last_data = (torch.inf, 0.0, torch.inf, 0.0)

    for e in range(epochs):
        print()
        print(Text("Epoch %d" % (e + 1,), justify = "center", style = "bold"))
        print()
        start_time = time.time()

        model, train_loss, train_acc = train_one_epoch(
            model = model,
            data_loader = train_data_loader,
            num_classes = num_classes,
            optimizer = optimizer,
            device = device
        )

        model, test_loss, test_acc = test_one_epoch(
            model = model,
            data_loader = test_data_loader,
            num_classes = num_classes,
            device = device,
            dtype = dtype
        )
        
        end_time = time.time()
        duration = end_time - start_time

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_model(log_dir, "best", model)
        
        current_data = (train_loss, train_acc, test_loss, test_acc)
        print_result(
            ep = e,
            data = current_data,
            last_data = last_data,
            max_test_acc = max_test_acc,
            duration = duration
        )
        last_data = current_data
        with open(os.path.join(log_dir, "result.csv"), "a") as f:
            f.write("%d, %g, %g, %g, %g, %g\n" % (e, train_loss, train_acc, test_loss, test_acc, duration))

        if scheduler is not None:
            scheduler.step()
        
        save_model(log_dir, "last", model)