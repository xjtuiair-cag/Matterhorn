import time
import torch
import torch.nn as nn
from matterhorn.snn import container, soma
from numba import jit

def main():
    model = container.Temporal(
        container.Spatial(
            nn.Linear(128, 80),
            soma.LIF(),
            nn.Linear(80, 10),
            soma.LIF()
        )
    )
    data = torch.Tensor(16, 4, 128)
    out = model(data)
    print(out.shape)


def main2():
    start = time.time()
    for i in range(100000):
        for j in range(3000):
            c = i + j
    end = time.time()
    print(end - start)

    start = time.time()
    for i in range(3000):
        for j in range(100000):
            c = i + j
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main2()