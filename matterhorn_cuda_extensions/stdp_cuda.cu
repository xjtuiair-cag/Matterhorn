#include <stdlib.h>
#include <cmath>
#include <iostream>
#include "base.h"
#include "stdp.h"

/*
STDP主函数。
Args:
    weight_mat (at::Tensor): 待更新的权重矩阵，形状为[output_shape,input_shape]
    input_shape (int): 输入向量长度
    output_shape (int): 输出向量长度
    time_steps (int): 时间步长
    input_spike_train (at::Tensor): 输入脉冲序列，形状为[time_steps,input_shape]
    output_spike_train (at::Tensor):
输出脉冲序列，形状为[time_steps,output_shape] a_pos (float): STDP参数$A^{+}$
    tau_pos (float): STDP参数$τ^{+}$
    a_neg (float): STDP参数$A^{-}$
    tau_neg (float): STDP参数$τ_{-}$
*/
__global__ void stdp_cuda_kernel(float* weight_mat,
                                 int input_shape,
                                 int output_shape,
                                 int time_steps,
                                 float* input_spike_train,
                                 float* output_spike_train,
                                 float a_pos,
                                 float tau_pos,
                                 float a_neg,
                                 float tau_neg,
                                 int batch_size = 1) {
    // 待更新的是W_{ij}
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= output_shape || j >= input_shape) {
        return;
    }

    // 去遍历时间，更新权重
    float weight = 0.0f;
    // 遍历输出脉冲
    for (int b = 0; b < batch_size; b++) {
        int wi = batch_size * output_shape;
        int bi = b * output_shape + i;
        int wj = batch_size * input_shape;
        int bj = b * input_shape + j;
        for (int ti = 0; ti < time_steps; ti++) {
            float spike_i = output_spike_train[ti * wi + bi];
            if (spike_i < 0.5f) {
                continue;
            }
            // 遍历输入脉冲
            for (int tj = 0; tj < time_steps; tj++) {
                float spike_j =
                    input_spike_train[tj * wj + bj];
                if (spike_j < 0.5f) {
                    continue;
                }
                float dt = (float)(ti - tj);
                if (dt > 0.0f) {
                    weight += a_pos * powf(1.0f / tau_pos, dt - 1.0f);
                } else if (dt < 0.0f) {
                    weight -= a_neg * powf(1.0f / tau_neg, -dt - 1.0f);
                }
            }
        }
    }
    weight_mat[i * input_shape + j] += weight;
}

void stdp_cuda(float* weight_mat,
               int input_shape,
               int output_shape,
               int time_steps,
               float* input_spike_train,
               float* output_spike_train,
               float a_pos,
               float tau_pos,
               float a_neg,
               float tau_neg,
               int batch_size = 1) {
    cudaError_t err;

    // i = blockIdx.y 为行，大小为output_shape
    // j = blockIdx.x * blockDim.x + threadIdx.x 为列，大小为input_shape
    dim3 blocks(div_ceil(input_shape, THREADS_PER_BLOCK), output_shape);
    dim3 threads(THREADS_PER_BLOCK);

    // 调用CUDA核心开始计算
    stdp_cuda_kernel<<<blocks, threads, 0>>>(
        weight_mat, input_shape, output_shape, time_steps, input_spike_train,
        output_spike_train, a_pos, tau_pos, a_neg, tau_neg, batch_size);

    // 返回计算结果
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}