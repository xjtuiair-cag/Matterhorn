import time
import torch
from matterhorn.snn import soma


def main():
    model = soma.LIF()
    a = torch.tensor([1., 2., 3., 4.])
    b = model(a)
    print(a, b)


if __name__ == "__main__":
    main()