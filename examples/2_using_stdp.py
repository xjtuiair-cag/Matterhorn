import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import matterhorn_pytorch.snn as snn
from torchvision.datasets.mnist import MNIST
from functions import *
from rich import print


def main():
    print_title("Example 2", style = "bold blue")

    print_title("Hyper Parameters")

    time_steps = 128
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    learning_rate = 1e-3
    momentum = 0.9
    tau = 2.0
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
        snn.PoissonEncoder(
            time_steps = 32
        ),
        snn.STDPConv2d(
            soma = snn.LIF(
                tau_m = tau
            ),
            in_channels = 1,
            out_channels = 4,
            kernel_size = 3,
            stride = 2,
            padding = 1
        ),
        snn.Flatten(),
        snn.Linear(784, 10, bias = False),
        snn.LIF(),
        snn.AvgSpikeDecoder()
    )
    model = model.to(device)
    print_model(model)

    print_title("Dataset")

    train_dataset = MNIST(
        root = "./examples/data",
        train = True,
        transform = torchvision.transforms.ToTensor(),
        download = True
    )
    test_dataset = MNIST(
        root = "./examples/data",
        train = False,
        transform = torchvision.transforms.ToTensor(),
        download = True
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
    print_dataset(test_dataset)
 
    print_title("Preparations for Training")

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = epochs)
    log_dir = "./examples/logs"
    sub_dir = "2_stdp" + "_" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(log_dir, sub_dir)
    init_logs(
        log_dir = log_dir,
        model = model
    )

    print_title("Training")

    train_and_test(
        epochs = epochs,
        model = model,
        train_data_loader = train_data_loader,
        test_data_loader = test_data_loader,
        num_classes = 10,
        optimizer = optimizer,
        scheduler = lr_scheduler,
        log_dir = log_dir,
        device = device
    )


if __name__ == "__main__":
    main()