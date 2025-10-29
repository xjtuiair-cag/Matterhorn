import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matterhorn_pytorch as mth
import matterhorn_pytorch.snn as snn
from matterhorn_pytorch.data import NMNIST
import argparse
from functions import *
from rich import print


def main():
    print_title("Example 3", style = "bold blue")

    print_title("Hyper Parameters")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type = str, default = "./examples/data", help = "Data path.")
    parser.add_argument("--logs-path", type = str, default = "./examples/logs", help = "Logs path.")
    parser.add_argument("--time-steps", type = int, default = 16, help = "Time steps.")
    parser.add_argument("--batch-size", type = int, default = 8, help = "Batch size.")
    parser.add_argument("--epochs", type = int, default = 100, help = "Training epochs.")
    parser.add_argument("--learning-rate", type = float, default = 0.01, help = "Learning rate.")
    parser.add_argument("--momentum", type = float, default = 0.9, help = "Momentum for optimizer.")
    parser.add_argument("--tau-m", type = float, default = 2.0, help = "Membrane constant.")

    args = parser.parse_args()
    time_steps = args.time_steps
    batch_size = args.batch_size
    device = get_proper_device()
    dtype = torch.float
    epochs = args.epochs
    learning_rate = args.learning_rate
    momentum = args.momentum
    tau = args.tau_m

    print_params({
        "Time Steps": time_steps,
        "Batch Size": batch_size,
        "Epochs": epochs,
        "Learning Rate": learning_rate,
        "Momentum": momentum,
        "Tau m": tau
    })

    print_title("Model")

    model = snn.Sequential(
        snn.DirectEncoder(),
        snn.Conv2d(
            in_channels = 2,
            out_channels = 32,
            kernel_size = 3,
            stride = 2,
            padding = 0
        ), # [T, B, 32, 16, 16]
        snn.BatchNorm2d(
            num_features = 32
        ),
        snn.LIF(
            tau_m = tau
        ),
        snn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            stride = 2,
            padding = 1
        ), # [T, B, 64, 8, 8]
        snn.BatchNorm2d(
            num_features = 64
        ),
        snn.LIF(
            tau_m = tau
        ),
        snn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            stride = 2,
            padding = 1
        ), # [T, B, 128, 4, 4]
        snn.BatchNorm2d(
            num_features = 128
        ),
        snn.LIF(
            tau_m = tau
        ),
        snn.Flatten(),
        snn.Linear(
            in_features = 128 * 4 * 4,
            out_features = 10,
            bias = False
        ),
        snn.AvgSpikeDecoder()
    )
    model = model.to(device)
    print_model(model)

    print_title("Dataset")

    train_dataset = NMNIST(
        root = args.data_path,
        train = True,
        download = True,
        cached = False,
        time_steps = time_steps
    )
    test_dataset = NMNIST(
        root = args.data_path,
        train = False,
        download = True,
        cached = False,
        time_steps = time_steps
    )
    train_data_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        num_workers = min(batch_size, max_workers()),
        shuffle = True,
        drop_last = True,
        pin_memory = True
    )
    test_data_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        num_workers = min(batch_size, max_workers()),
        shuffle = True,
        drop_last = True,
        pin_memory = True
    )
    print_dataset(test_dataset)

    print_title("Preparations for Training")

    def loss_fn(o: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(o.float(), y.long())
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = epochs)
    log_dir = args.logs_path
    sub_dir = "3_conv" + "_" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(log_dir, sub_dir)
    init_logs(
        log_dir = log_dir,
        args = args,
        model = model
    )
    
    print_title("Training")

    train_and_test(
        epochs = epochs,
        model = model,
        train_data_loader = train_data_loader,
        test_data_loader = test_data_loader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        scheduler = lr_scheduler,
        log_dir = log_dir,
        device = device,
        dtype = dtype
    )


if __name__ == "__main__":
    main()