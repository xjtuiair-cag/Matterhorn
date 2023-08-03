import torch
import torch.nn as nn
from rich import print


@torch.jit.script
def stdp_py(weight_mat: torch.Tensor, input_shape: int, output_shape: int, time_steps: int, input_spike_train: torch.Tensor, output_spike_train: torch.Tensor, a_pos: float, tau_pos: float, a_neg: float, tau_neg: float):
    for i in range(output_shape):
        for j in range(input_shape):
            weight = 0.0
            for ti in range(time_steps):
                if not output_spike_train[ti, i]:
                    continue
                for tj in range(time_steps):
                    if not input_spike_train[tj, j]:
                        continue
                    dt = ti - tj
                    if dt > 0:
                        weight += a_pos * torch.exp(-dt / tau_pos)
                    else:
                        weight += -a_neg * torch.exp(dt / tau_neg)
            weight_mat[i, j] += weight


if torch.cuda.is_available():
    try:
        from matterhorn_cuda import stdp
    except:
        stdp = stdp_py
else:
    stdp = stdp_py


if __name__ == "__main__":
    torch.random.manual_seed(2023)
    input_shape = 150
    output_shape = 10
    time_steps = 16


    input_spike_train = torch.rand(time_steps, input_shape).gt(0.7).float().cuda()
    output_spike_train = torch.rand(time_steps, output_shape).gt(0.7).float().cuda()
    weight_mat = torch.zeros(output_shape, input_shape).float().cuda()

    print(input_spike_train)
    print(input_spike_train.shape)
    print(output_spike_train)
    print(output_spike_train.shape)
    stdp(weight_mat, input_shape, output_shape, time_steps, input_spike_train, output_spike_train, 0.5, 4, 0.5, 4)
    print(weight_mat)
    print(weight_mat.shape)