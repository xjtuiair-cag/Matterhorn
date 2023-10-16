#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"
#include "stdp.h"

/*
LIF神经元的前向传播函数。
LIF神经元反应函数：
$$U_{i}^{l}(t)=H_{i}^{l}(t-1)+\frac{1}{τ_{m}}[-[H_{i}^{l}(t-1)-u_{rest}]+X_{i}^{l}(t)]$$
Heaviside脉冲函数：
$$O_{i}^{l}(t)=u[U_{i}^{l}(t)]$$
硬重置：
$$H_{i}^{l}(t)=U_{i}^{l}(t)[1-O_{i}^{l}(t)]+u_{rest}O_{i}^{l}(t)$$
@params:
    o: at::Tensor 脉冲输出$O^{l}$
    u: at::Tensor 胞体电位$U^{l}$
    h: at::Tensor 胞体历史电位$H^{l}$
    x: at::Tensor 输入电位$X^{l}$
    time_steps: int 总时间步长
    shape: int 总空间大小
    u_init: at::Tensor 初始胞体电位$H^{l}(-1)$
    tau_m: at::Tensor 时间常数$τ_{m}$
    u_rest: float 静息电位$u_{rest}$
    u_threshold: float 阈电位$u_{th}$
*/
__global__ void p_lif_heaviside_hard_cuda(float* o,
                              float* u,
                              float* h,
                              const float* x,
                              int time_steps,
                              int shape,
                              const float* u_init,
                              const float* tau_m,
                              float u_rest,
                              float u_threshold) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * 1024 + j;
    if (idx >= shape) {
        return;
    }
    
    for (int t = 0; t < time_steps; t++) {
        // $$U_{i}^{l}(t)=H_{i}^{l}(t-1)+\frac{1}{τ_{m}}[-[H_{i}^{l}(t-1)-u_{rest}]+X_{i}^{l}(t)]$$
        float h_t_1 = t ? h[(t - 1) * shape + i] : u_init[i];
        u[t * shape + i] = h_t_1 + (1 / tau_m[0]) * (-h_t_1 - u_rest + x[t * shape + i]);
        // $$O_{i}^{l}(t)=u[U_{i}^{l}(t)]$$
        o[t * shape + i] = u[t * shape + i] >= u_threshold ? 1.0 : 0.0;
        // $$H_{i}^{l}(t)=U_{i}^{l}(t)[1-O_{i}^{l}(t)]+u_{rest}O_{i}^{l}(t)$$
        h[t * shape + i] = u[t * shape + i] * (1 - o[t * shape + i]) + u_rest * o[t * shape + i];
    }
}

void p_lif_heaviside_hard_cuda(float* o,
                              float* u,
                              float* h,
                              const float* x,
                              int time_steps,
                              int shape,
                              const float* u_init,
                              const float* tau_m,
                              float u_rest,
                              float u_threshold) {
    cudaError_t err;

    // i = blockIdx.y 为行
    // j = blockIdx.x * blockDim.x + threadIdx.x 为列
    dim3 blocks(DIVUP(DIVUP(shape, 1024), THREADS_PER_BLOCK), 1024);
    dim3 threads(THREADS_PER_BLOCK);

    // 调用CUDA核心开始计算
    p_lif_heaviside_hard_cuda<<<blocks, threads, 0>>>(o, u, h, x, time_steps, shape, u_init, tau_m, u_rest, u_threshold);

    // 返回计算结果
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}