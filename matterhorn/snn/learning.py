import torch
import torch.nn as nn
from rich import print
from matterhorn_cuda import stdp


if __name__ == "__main__":
    input_shape = 128 * 128 * 2
    output_shape = 800
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