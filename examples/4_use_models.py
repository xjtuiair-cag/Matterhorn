import torch
from torch.utils.data import DataLoader
import matterhorn_pytorch as mth
from matterhorn_pytorch.model.sew import SEWRes18
from matterhorn_pytorch.data import CIFAR10DVS
import argparse
from functions import *
from rich import print

 
def main():
    print_title("Example 4", style = "bold blue")

    print_title("Hyper Parameters")

    parser = argparse.ArgumentParser()
    parser.add_argument("--time-steps", type = int, default = 32, help = "Time steps.")
    parser.add_argument("--batch-size", type = int, default = 8, help = "Batch size.")
    parser.add_argument("--device", type = str, default = "cpu", help = "Device for running the models.")
    parser.add_argument("--epochs", type = int, default = 100, help = "Training epochs.")
    parser.add_argument("--learning-rate", type = float, default = 0.01, help = "Learning rate.")
    parser.add_argument("--momentum", type = float, default = 0.9, help = "Momentum for optimizer.")
    parser.add_argument("--tau-m", type = float, default = 2.0, help = "Membrane constant.")

    args = parser.parse_args()
    time_steps = args.time_steps
    batch_size = args.batch_size
    device = torch.device(args.device)
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

    model = SEWRes18(
        input_h_w = (128, 128),
        loss_fn = loss_fn,
        tau_m = tau
    )
    model.multi_step_mode_()
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

    def loss_fn(o: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(o.float(), y.float())
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = epochs)
    log_dir = "./examples/logs"
    sub_dir = "4_model" + "_" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
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
        device = device
    )


if __name__ == "__main__":
    main()