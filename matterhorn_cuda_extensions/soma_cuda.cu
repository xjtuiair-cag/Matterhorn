#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"
#include "soma.h"

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
__global__ void fp_lif_heaviside_hard_cuda_kernel(float* o,
                                                  float* u,
                                                  float* h,
                                                  float* x,
                                                  int time_steps,
                                                  int shape,
                                                  float* u_init,
                                                  float* tau_m,
                                                  float u_rest,
                                                  float u_threshold) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * 1024 + j;
    if (idx >= shape) {
        return;
    }

    const float tau_m_val = tau_m[0];
    for (int t = 0; t < time_steps; t++) {
        const int cur_idx = t * shape + idx;
        float last_h = t ? h[cur_idx - shape] : u_init[idx];
        
        u[cur_idx] = last_h + (1.0f / tau_m_val) * (0.0f - (last_h - u_rest) + x[cur_idx]);
        
        o[cur_idx] = u[cur_idx] >= u_threshold ? 1.0f : 0.0f;
        
        h[cur_idx] = u[cur_idx] * (1.0f - o[cur_idx]) + u_rest * o[cur_idx];
    }
}

void fp_lif_heaviside_hard_cuda(float* o,
                                float* u,
                                float* h,
                                float* x,
                                int time_steps,
                                int shape,
                                float* u_init,
                                float* tau_m,
                                float u_rest,
                                float u_threshold) {
    cudaError_t err;

    // i = blockIdx.y 为行
    // j = blockIdx.x * blockDim.x + threadIdx.x 为列
    dim3 blocks(DIVUP(DIVUP(shape, 1024), THREADS_PER_BLOCK), 1024);
    dim3 threads(THREADS_PER_BLOCK);

    // 调用CUDA核心开始计算
    fp_lif_heaviside_hard_cuda_kernel<<<blocks, threads, 0>>>(
        o, u, h, x, time_steps, shape, u_init, tau_m, u_rest, u_threshold);

    // 返回计算结果
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

/*
LIF神经元的反向传播函数。
LIF神经元反应函数的反向传播：
$$\frac{\partial U_{i}^{l}(t)}{\partial H_{i}^{l}(t-1)}=1-\frac{1}{τ_{m}}$$
$$\frac{\partial U_{i}^{l}(t)}{\partial X_{i}^{l}(t)}=\frac{1}{τ_{m}}$$
$$\frac{\partial U_{i}^{l}(t)}{\partial
τ_{m}}=-\frac{1}{τ_{m}^{2}}[-[H_{i}^{l}(t-1)-u_{rest}]+X_{i}^{l}(t)]$$
矩形窗反向传播：
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=u'$$
硬重置反向传播：
$$\frac{\partial H_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=1-O_{i}^{l}(t)$$
$$\frac{\partial H_{i}^{l}(t)}{\partial O_{i}^{l}(t)}=-U_{i}^{l}(t)+u_{rest}$$
@params:
    grad_o: at::Tensor 脉冲输出$O^{l}$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}$的梯度
    grad_h: at::Tensor 胞体历史电位$H^{l}$的梯度
    grad_x: at::Tensor 输入电位$X^{l}$的梯度
    grad_u_init: at::Tensor 初始胞体电位$H^{l}(-1)$的梯度
    grad_tau_m: at::Tensor 时间常数$τ_{m}$的梯度
    time_steps: int 总时间步长
    shape: int 总空间大小
    o: at::Tensor 脉冲输出$O^{l}$
    u: at::Tensor 胞体电位$U^{l}$
    h: at::Tensor 胞体历史电位$H^{l}$
    x: at::Tensor 输入电位$X^{l}$
    u_init: at::Tensor 初始胞体电位$H^{l}(-1)$
    tau_m: at::Tensor 时间常数$τ_{m}$
    u_rest: float 静息电位$u_{rest}$
    u_threshold: float 阈电位$u_{th}$
*/
__global__ void bp_lif_rectangular_hard_cuda_kernel(float* grad_o,
                                                    float* grad_u,
                                                    float* grad_h,
                                                    float* grad_x,
                                                    float* grad_u_init,
                                                    float* grad_tau_m,
                                                    int time_steps,
                                                    int shape,
                                                    float* o,
                                                    float* u,
                                                    float* h,
                                                    float* x,
                                                    float* u_init,
                                                    float* tau_m,
                                                    float u_rest,
                                                    float u_threshold) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * 1024 + j;
    if (idx >= shape) {
        return;
    }

    const float tau_m_val = tau_m[0];
    float cur_grad_h = 0.0f;
    for (int t = time_steps - 1; t >= 0; t--) {
        const int cur_idx = t * shape + idx;
        float last_h = t ? h[cur_idx - shape] : u_init[idx];

        grad_u[cur_idx] += cur_grad_h * (1.0f - o[cur_idx]);
        grad_o[cur_idx] += cur_grad_h * (0.0f - u[cur_idx] + u_rest);

        grad_u[cur_idx] += grad_o[cur_idx] * 0.5f * (((u[cur_idx] > (u_threshold - 1.0f)) && (u[cur_idx] < (u_threshold + 1.0f))) ? 1.0f : 0.0f);
        
        grad_x[cur_idx] += grad_u[cur_idx] * (1.0f / tau_m_val);
        cur_grad_h = grad_u[cur_idx] * (1.0f - (1.0f / tau_m_val));
        grad_tau_m[0] += grad_u[cur_idx] * (0.0f - (1.0f / (tau_m_val * tau_m_val)) * (0.0f - (last_h - u_rest) + x[cur_idx]));
        if (t) {
            grad_h[cur_idx - shape] = cur_grad_h;
        } else {
            grad_u_init[idx] = cur_grad_h;
        }
    }
}

void bp_lif_rectangular_hard_cuda(float* grad_o,
                                  float* grad_u,
                                  float* grad_h,
                                  float* grad_x,
                                  float* grad_u_init,
                                  float* grad_tau_m,
                                  int time_steps,
                                  int shape,
                                  float* o,
                                  float* u,
                                  float* h,
                                  float* x,
                                  float* u_init,
                                  float* tau_m,
                                  float u_rest,
                                  float u_threshold) {
    cudaError_t err;

    // i = blockIdx.y 为行
    // j = blockIdx.x * blockDim.x + threadIdx.x 为列
    dim3 blocks(DIVUP(DIVUP(shape, 1024), THREADS_PER_BLOCK), 1024);
    dim3 threads(THREADS_PER_BLOCK);

    // 调用CUDA核心开始计算
    bp_lif_rectangular_hard_cuda_kernel<<<blocks, threads, 0>>>(
        grad_o, grad_u, grad_h, grad_x, grad_u_init, grad_tau_m, time_steps,
        shape, o, u, h, x, u_init, tau_m, u_rest, u_threshold);

    // 返回计算结果
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}