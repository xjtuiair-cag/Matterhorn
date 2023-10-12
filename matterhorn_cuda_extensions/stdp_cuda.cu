#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"
#include "stdp.h"

/*
STDP的核心代码，利用CUDA实现。
@params:
    weight_mat: float* 待更新的权重矩阵，形状为[output_shape,input_shape]
    input_shape: int 输入向量长度
    output_shape: int 输出向量长度
    time_steps: int 时间步长
    input_spike_train: float* 输入脉冲序列，形状为[time_steps,input_shape]
    output_spike_train: float* 输出脉冲序列，形状为[time_steps,output_shape]
    a_pos: float STDP参数$A^{+}$
    tau_pos: float STDP参数$τ^{+}$
    a_neg: float STDP参数$A^{-}$
    tau_neg: float STDP参数$τ_{-}$
*/
__global__ void stdp_cuda_kernel(float* weight_mat,
                                 int input_shape,
                                 int output_shape,
                                 int time_steps,
                                 const float* input_spike_train,
                                 const float* output_spike_train,
                                 float a_pos,
                                 float tau_pos,
                                 float a_neg,
                                 float tau_neg) {
    // 待更新的是W_{ij}
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= output_shape || j >= input_shape) {
        return;
    }

    // 去遍历时间，更新权重
    float weight = 0.0f;
    // 遍历输出脉冲
    for (int ti = 0; ti < time_steps; ti++) {
        float spike_i = output_spike_train[i + ti * output_shape];
        if (!spike_i) {
            continue;
        }
        // 遍历输入脉冲
        for (int tj = 0; tj < time_steps; tj++) {
            float spike_j = input_spike_train[j + tj * input_shape];
            if (!spike_j) {
                continue;
            }
            int dt = ti - tj;
            if (dt > 0) {
                weight += a_pos * expf(-dt / tau_pos);
            } else if (dt < 0) {
                weight += -a_neg * expf(dt / tau_neg);
            }
        }
    }
    weight_mat[i * input_shape + j] += weight;
}

/*
调用CUDA的STDP函数。
@params:
    weight_mat: float* 待更新的权重矩阵，形状为[output_shape, input_shape]
    input_shape: int 输入向量长度
    output_shape: int 输出向量长度
    time_steps: int 时间步长
    input_spike_train: float* 输入脉冲序列，形状为[time_steps, input_shape]
    output_spike_train: float* 输出脉冲序列，形状为[time_steps, output_shape]
    a_pos: float STDP参数A+
    tau_pos: float STDP参数tau+
    a_neg: float STDP参数A-
    tau_neg: float STDP参数tau-
    stream: cudaStream_t CUDA流
*/
void stdp_cuda(float* weight_mat,
               int input_shape,
               int output_shape,
               int time_steps,
               const float* input_spike_train,
               const float* output_spike_train,
               float a_pos,
               float tau_pos,
               float a_neg,
               float tau_neg) {
    cudaError_t err;

    // i = blockIdx.y 为行，大小为output_shape
    // j = blockIdx.x * blockDim.x + threadIdx.x 为列，大小为input_shape
    dim3 blocks(DIVUP(input_shape, THREADS_PER_BLOCK), output_shape);
    dim3 threads(THREADS_PER_BLOCK);

    // 调用CUDA核心开始计算
    stdp_cuda_kernel<<<blocks, threads, 0>>>(
        weight_mat, input_shape, output_shape, time_steps, input_spike_train,
        output_spike_train, a_pos, tau_pos, a_neg, tau_neg);

    // 返回计算结果
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}