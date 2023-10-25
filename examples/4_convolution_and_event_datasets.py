import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import time
import os, sys
sys.path.append(os.path.abspath("."))


import matterhorn_pytorch
import matterhorn_pytorch.snn as snn


from rich import print
from rich.panel import Panel
from rich.text import Text
from rich.progress import track
from rich.table import Table


def main():
    # 欢迎语，客套话

    print(Panel(Text("EXAMPLE 4: CONVOLUTIONAL SPIKING NEURAL NETWORKS", justify = "center", style = "bold blue")))

    print("This is your fourth example. You'll learn to use event datasets and convolution operation on [green]Matterhorn[/green].")

    print("In this demo, we're about to build a 3-layer convolutional network.")

    # 设置参数

    print(Panel(Text("Hyper Parameters", justify = "center")))

    time_steps = 128
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 32
    learning_rate = 1e-3
    momentum = 0.9
    tau = 1.1

    hyper_param_table = Table(show_header = True, header_style = "bold blue")
    hyper_param_table.add_column("Name", justify = "center")
    hyper_param_table.add_column("Value", justify = "center")
    hyper_param_table.add_row("Time Steps", str(time_steps))
    hyper_param_table.add_row("Batch Size", str(batch_size))
    hyper_param_table.add_row("Epochs", str(epochs))
    hyper_param_table.add_row("Learning Rate", str(learning_rate))
    hyper_param_table.add_row("Momentum", str(momentum))
    hyper_param_table.add_row("Tau m", str(tau))
    print(hyper_param_table)

    # 根据参数建立模型

    print(Panel(Text("Model", justify = "center")))

    model = snn.Sequential(
        snn.DirectEncoder(),
        snn.Conv2d(in_channels = 2, out_channels = 64, kernel_size = 3, stride = 2, padding = 1), # [T, 64, 17, 17]
        snn.LIF(tau_m = tau),
        snn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1), # [T, 64, 9, 9]
        snn.LIF(tau_m = tau),
        snn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1), # [T, 64, 5, 5]
        snn.LIF(tau_m = tau),
        snn.AvgSpikeDecoder(),
        nn.Flatten(),
        nn.Linear(1600, 10),
        nn.ReLU()
    )
    model = model.to(device)

    print(model)
    print(device)

    # 调取数据集，本次使用的数据集为MNIST

    print(Panel(Text("Dataset", justify = "center")))

    width = 34
    height = 34
    train_dataset = matterhorn_pytorch.data.NMNIST(
        root = "./examples/data",
        train = True,
        download = True,
        time_steps = time_steps,
        width = width,
        height = height
    )
    test_dataset = matterhorn_pytorch.data.NMNIST(
        root = "./examples/data",
        train = False,
        download = True,
        time_steps = time_steps,
        width = width,
        height = height
    )

    train_data_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        pin_memory = True
    )
    test_data_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        pin_memory = True
    )

    demo_data, demo_label = test_dataset[0]
    print(demo_data.shape)
    # matterhorn_pytorch.util.plotter.event_plot_tyx(demo_data, titles = ["%s Label %s" % (test_dataset.__class__.__name__, test_dataset.labels[demo_label])])

    # 设置学习率，优化器，学习率衰减机制等等

    print(Panel(Text("Preparations for Training", justify = "center")))

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = epochs)

    # 开始训练

    print(Panel(Text("Training", justify = "center")))

    max_test_acc = 0.0
    last_train_loss = torch.inf
    last_train_acc = 0.0
    last_test_loss = torch.inf
    last_test_acc = 0.0
    get_color = lambda x, y: "green" if x > y else ("red" if x < y else "blue")

    for e in range(epochs):
        start_time = time.time()

        # 使用训练集进行训练

        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_samples = 0
        for x, y in track(train_data_loader, description = "Training at epoch %d" % (e,)):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            y0 = torch.nn.functional.one_hot(y, num_classes = len(train_dataset.labels)).float()

            o = model(x)
            loss = torch.nn.functional.mse_loss(o, y0)
            loss.backward()
            optimizer.step()

            train_samples += y.numel()
            train_loss += loss.item() * y.numel()
            train_acc += (o.argmax(1) == y).float().sum().item()

        train_loss /= train_samples
        train_acc /= train_samples
        
        # 使用测试集进行评估

        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_samples = 0
        with torch.no_grad():
            for x, y in track(test_data_loader, description = "Testing at epoch %d" % (e,)):
                x = x.to(device)
                y = y.to(device)
                y0 = torch.nn.functional.one_hot(y, num_classes = len(train_dataset.labels)).float()
                
                o = model(x)
                loss = torch.nn.functional.mse_loss(o, y0)

                test_samples += y.numel()
                test_loss += loss.item() * y.numel()
                test_acc += (o.argmax(1) == y).float().sum().item()
        
        test_loss /= test_samples
        test_acc /= test_samples
        if test_acc > max_test_acc:
            max_test_acc = test_acc
        
        end_time = time.time()

        # 打印测试结果

        result_table = Table(show_header = True, header_style = "bold blue")
        result_table.add_column("Name", justify = "center")
        result_table.add_column("Value", justify = "center")
        result_table.add_row("Epoch", str(e))
        result_table.add_row("Learning Rate", "%g" % (lr_scheduler.get_last_lr()[0],))
        result_table.add_row("Training Loss", "[%s]%g[/%s]" % (get_color(last_train_loss, train_loss), train_loss, get_color(last_train_loss, train_loss)))
        result_table.add_row("Training Accuracy", "[%s]%g%%[/%s]" % (get_color(train_acc, last_train_acc), 100 * train_acc,get_color(train_acc, last_train_acc)))
        result_table.add_row("Testing Loss", "[%s]%g[/%s]" % (get_color(last_test_loss, test_loss), test_loss, get_color(last_test_loss, test_loss)))
        result_table.add_row("Testing Accuracy", "[%s]%g%%[/%s]" % (get_color(test_acc, last_test_acc), 100 * test_acc, get_color(test_acc, last_test_acc)))
        result_table.add_row("Maximum Testing Accuracy", "%g%%" % (100 * max_test_acc,))
        result_table.add_row("Duration", "%gs" %(end_time - start_time,))
        print(result_table)

        last_train_loss = train_loss
        last_train_acc = train_acc
        last_test_loss = test_loss
        last_test_acc = test_acc
        lr_scheduler.step()

if __name__ == "__main__":
    main()