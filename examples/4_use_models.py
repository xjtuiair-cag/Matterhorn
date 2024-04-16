import torch
from torch.utils.data import DataLoader
import matterhorn_pytorch as mth
from matterhorn_pytorch.model.sew import SEWRes18
from matterhorn_pytorch.data import CIFAR10DVS
from functions import *
from rich import print

 
def main():
    print_title("Example 4", style = "bold blue")

    print_title("Hyper Parameters")

    time_steps = 32
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 64
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

    model = SEWRes18(
        input_h_w = (128, 128),
        num_classes = 10,
        tau_m = tau
    )
    model = model.to(device)
    print_model(model)

    print_title("Dataset")

    train_dataset = CIFAR10DVS(
        root = "./examples/data",
        train = True,
        download = True,
        time_steps = time_steps
    )
    test_dataset = CIFAR10DVS(
        root = "./examples/data",
        train = False,
        download = True,
        time_steps = time_steps
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
    sub_dir = "4_model" + "_" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
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