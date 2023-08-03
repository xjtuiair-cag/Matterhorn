#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"
#include "stdp.h"

__global__ void stdp_cuda_kernel(const float* __restrict__* weight_mat,
                                 int input_shape,
                                 int output_shape,
                                 int time_steps,
                                 const float* __restrict__* input_spike_train,
                                 const float* __restrict__* output_spike_train,
                                 float a_pos,
                                 float tau_pos,
                                 float a_neg,
                                 float tau_neg, ) {
    // 待更新的是W_{ij}
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= output_shape || j >= input_shape)
        return;
    
    float weight = 0.0f;
    for (int ti = 0; ti < time_steps; ti++) {
        float spike_i = output_spike_train[i + ti * input_shape];
        if (!spike_i) {
            continue;
        }
        for (int tj = 0; tj < time_steps; tj++) {
            float spike_j = input_spike_train[j + tj * output_shape];
            if (!spike_j) {
                continue;
            }
            int dt = ti - tj;
            if (dt > 0) {
                weight = a_pos * expf32(-dt / tau_pos);
            } else if (dt < 0) {
                weight = a_neg * expf32(dt / tau_neg);
            }
        }
    }
    weight_mat[i * output_shape + j] += weight;
}

void stdp_cuda(const float* weight_mat,
               int input_shape,
               int output_shape,
               int time_steps,
               const float* input_spike_train,
               const float* output_spike_train,
               float a_pos,
               float tau_pos,
               float a_neg,
               float tau_neg,
               cudaStream_t stream) {
    cudaError_t err;

    dim3 blocks(DIVUP(input_shape, THREADS_PER_BLOCK),
                output_shape);  // blockIdx.x(col=input_shape),
                                // blockIdx.y(row=output_shape)
    dim3 threads(THREADS_PER_BLOCK);

    stdp_cuda_kernel<<<blocks, threads, 0, stream>>>(
        weight_mat, input_shape, output_shape, time_steps, input_spike_train,
        output_spike_train, a_pos, tau_pos, a_neg, tau_neg);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}