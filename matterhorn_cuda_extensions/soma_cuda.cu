#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"
#include "soma.h"

#define SURROGATE_RECTANGULAR 0
#define SURROGATE_POLYNOMIAL 1
#define SURROGATE_SIGMOID 2
#define SURROGATE_GAUSSIAN 3

#define RESET_HARD 0
#define RESET_SOFT 1

/*
LIF神经元反应函数的前向传播函数。
$$U_{i}^{l}(t)=H_{i}^{l}(t-1)+\frac{1}{τ_{m}}[-[H_{i}^{l}(t-1)-u_{rest}]+X_{i}^{l}(t)]$$
@params:
    u: at::Tensor 胞体电位$U^{l}(t)$
    x: at::Tensor 输入电位$X^{l}(t)$
    h: at::Tensor 胞体历史电位$H^{l}(t-1)$
    tau_m: at::Tensor 时间常数$τ_{m}$
    u_rest: float 静息电位$u_{rest}$
*/
__device__ void fp_response_lif(float& u,
                                float x,
                                float h,
                                float tau_m,
                                float u_rest) {
    u = h + (1.0f / tau_m) * (0.0f - (h - u_rest) + x);
}

/*
LIF神经元反应函数的反向传播函数。
$$\frac{\partial U_{i}^{l}(t)}{\partial H_{i}^{l}(t-1)}=1-\frac{1}{τ_{m}}$$
$$\frac{\partial U_{i}^{l}(t)}{\partial X_{i}^{l}(t)}=\frac{1}{τ_{m}}$$
$$\frac{\partial U_{i}^{l}(t)}{\partial
τ_{m}}=-\frac{1}{τ_{m}^{2}}[-[H_{i}^{l}(t-1)-u_{rest}]+X_{i}^{l}(t)]$$
@params:
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    grad_x: at::Tensor 输入电位$X^{l}(t)$的梯度
    grad_h: at::Tensor 胞体历史电位$H^{l}(t-1)$的梯度
    grad_tau_m: at::Tensor 时间常数$τ_{m}$的梯度
    u: at::Tensor 胞体电位$U^{l}(t)$
    h: at::Tensor 胞体历史电位$H^{l}(t)$
    x: at::Tensor 输入电位$X^{l}(t-1)$
    tau_m: at::Tensor 时间常数$τ_{m}$
    u_rest: float 静息电位$u_{rest}$
*/
__device__ void bp_response_lif(float grad_u,
                                float& grad_x,
                                float& grad_h,
                                float& grad_tau_m,
                                float u,
                                float x,
                                float h,
                                float tau_m,
                                float u_rest) {
    grad_x += grad_u * (1.0f / tau_m);
    grad_h = grad_u * (1.0f - (1.0f / tau_m));
    grad_tau_m +=
        grad_u * (0.0f - (1.0f / (tau_m * tau_m)) * (0.0f - (h - u_rest) + x));
}

/*
Heaviside脉冲函数前向传播函数。
$$O_{i}^{l}(t)=u[U_{i}^{l}(t)]$$
@params:
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    u_threshold: float 阈电位$u_{th}$
*/
__device__ void fp_spiking_heaviside(float& o, float u, float u_threshold) {
    o = u >= u_threshold ? 1.0f : 0.0f;
}

/*
矩形窗反向传播函数。
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=u'$$
@params:
    grad_o: at::Tensor 脉冲输出$O^{l}(t)$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    u_threshold: float 阈电位$u_{th}$
    a: float 参数$a$
*/
__device__ void bp_spiking_rectangular(float grad_o,
                                       float& grad_u,
                                       float o,
                                       float u,
                                       float u_threshold,
                                       float a) {
    float ax = u - u_threshold;
    float rect = ((ax > -1.0f) && (ax < 1.0f)) ? 1.0f : 0.0f;
    grad_u += grad_o * 0.5f * rect;
}

/*
多项式反向传播函数。
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=u'$$
@params:
    grad_o: at::Tensor 脉冲输出$O^{l}(t)$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    u_threshold: float 阈电位$u_{th}$
    a: float 参数$a$
*/
__device__ void bp_spiking_polynomial(float grad_o,
                                      float& grad_u,
                                      float o,
                                      float u,
                                      float u_threshold,
                                      float a) {
    float ax = u - u_threshold;
    float sign =
        2.0f / sqrtf(a) != ax ? (2.0f / sqrtf(a) > ax ? 1.0f : -1.0f) : 0.0f;
    grad_u += grad_o * (sqrtf(a) / 2.0f - a / 4.0f * ax) * sign;
}

/*
Sigmoid导数的反向传播函数。
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=u'$$
@params:
    grad_o: at::Tensor 脉冲输出$O^{l}(t)$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    u_threshold: float 阈电位$u_{th}$
    a: float 参数$a$
*/
__device__ void bp_spiking_sigmoid(float grad_o,
                                   float& grad_u,
                                   float o,
                                   float u,
                                   float u_threshold,
                                   float a) {
    float ax = u - u_threshold;
    float ex = expf(-ax / a);
    grad_u += grad_o * (1.0f / a) * ex / powf(1.0f + ex, 2.0f);
}

/*
高斯反向传播函数。
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=u'$$
@params:
    grad_o: at::Tensor 脉冲输出$O^{l}(t)$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    u_threshold: float 阈电位$u_{th}$
    a: float 参数$a$
*/
__device__ void bp_spiking_gaussian(float grad_o,
                                    float& grad_u,
                                    float o,
                                    float u,
                                    float u_threshold,
                                    float a) {
    float ax = u - u_threshold;
    grad_u += grad_o * 1.0f / sqrtf(2.0f * M_PI * a) *
              expf(-powf(ax, 2.0f) / (2.0f * a));
}

/*
硬重置前向传播函数。
$$H_{i}^{l}(t)=U_{i}^{l}(t)[1-O_{i}^{l}(t)]+u_{rest}O_{i}^{l}(t)$$
@params:
    h: at::Tensor 胞体历史电位$H^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u_rest: float 静息电位$u_{rest}$
*/
__device__ void fp_reset_hard(float& h, float u, float o, float u_rest) {
    h = u * (1.0f - o) + u_rest * o;
}

/*
硬重置反向传播函数。
$$\frac{\partial H_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=1-O_{i}^{l}(t)$$
$$\frac{\partial H_{i}^{l}(t)}{\partial O_{i}^{l}(t)}=-U_{i}^{l}(t)+u_{rest}$$
@params:
    grad_h: at::Tensor 胞体历史电位$H^{l}(t)$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    grad_o: at::Tensor 脉冲输出$O^{l}(t)$的梯度
    h: at::Tensor 胞体历史电位$H^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u_rest: float 静息电位$u_{rest}$
*/
__device__ void bp_reset_hard(float grad_h,
                              float& grad_u,
                              float& grad_o,
                              float h,
                              float u,
                              float o,
                              float u_rest) {
    grad_u += grad_h * (1.0f - o);
    grad_o += grad_h * (u_rest - u);
}

/*
软重置前向传播函数。
$$H_{i}^{l}(t)=U_{i}^{l}(t)[1-O_{i}^{l}(t)]+u_{rest}O_{i}^{l}(t)$$
@params:
    h: at::Tensor 胞体历史电位$H^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u_threshold: float 阈电位$u_{th}$
    u_rest: float 静息电位$u_{rest}$
*/
__device__ void fp_reset_soft(float& h,
                              float u,
                              float o,
                              float u_threshold,
                              float u_rest) {
    h = u - (u_threshold - u_rest) * o;
}

/*
软重置反向传播函数。
$$\frac{\partial H_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=1-O_{i}^{l}(t)$$
$$\frac{\partial H_{i}^{l}(t)}{\partial O_{i}^{l}(t)}=-U_{i}^{l}(t)+u_{rest}$$
@params:
    grad_h: at::Tensor 胞体历史电位$H^{l}(t)$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    grad_o: at::Tensor 脉冲输出$O^{l}(t)$的梯度
    h: at::Tensor 胞体历史电位$H^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u_threshold: float 阈电位$u_{th}$
    u_rest: float 静息电位$u_{rest}$
*/
__device__ void bp_reset_soft(float grad_h,
                              float& grad_u,
                              float& grad_o,
                              float h,
                              float u,
                              float o,
                              float u_threshold,
                              float u_rest) {
    grad_u += grad_h * 1.0f;
    grad_o += grad_h * -(u_threshold - u_rest);
}

/*
LIF神经元的前向传播函数。
@params:
    o: at::Tensor 脉冲输出$O^{l}$
    u: at::Tensor 胞体电位$U^{l}$
    h: at::Tensor 胞体历史电位$H^{l}$
    x: at::Tensor 输入电位$X^{l}$
    time_steps: int 总时间步长
    u_init: at::Tensor 初始胞体电位$H^{l}(-1)$
    tau_m: at::Tensor 时间常数$τ_{m}$
    u_rest: float 静息电位$u_{rest}$
    u_threshold: float 阈电位$u_{th}$
    reset_mode: int 重置模式，分为硬重置（0）和软重置（1）两种
*/
__global__ void fp_lif_cuda_kernel(float* o,
                                   float* u,
                                   float* h,
                                   float* x,
                                   int time_steps,
                                   int shape,
                                   float* u_init,
                                   float* tau_m,
                                   float u_rest,
                                   float u_threshold,
                                   int reset_mode) {
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
        fp_response_lif(u[cur_idx], x[cur_idx], last_h, tau_m_val, u_rest);
        fp_spiking_heaviside(o[cur_idx], u[cur_idx], u_threshold);
        switch (reset_mode) {
            case RESET_HARD:
                fp_reset_hard(h[cur_idx], u[cur_idx], o[cur_idx], u_rest);
                break;
            case RESET_SOFT:
                fp_reset_soft(h[cur_idx], u[cur_idx], o[cur_idx], u_threshold,
                              u_rest);
                break;
        }
    }
}

void fp_lif_cuda(float* o,
                 float* u,
                 float* h,
                 float* x,
                 int time_steps,
                 int shape,
                 float* u_init,
                 float* tau_m,
                 float u_rest,
                 float u_threshold,
                 int reset_mode) {
    cudaError_t err;

    // i = blockIdx.y 为行
    // j = blockIdx.x * blockDim.x + threadIdx.x 为列
    dim3 blocks(DIVUP(DIVUP(shape, 1024), THREADS_PER_BLOCK), 1024);
    dim3 threads(THREADS_PER_BLOCK);

    // 调用CUDA核心开始计算
    fp_lif_cuda_kernel<<<blocks, threads, 0>>>(o, u, h, x, time_steps, shape,
                                               u_init, tau_m, u_rest,
                                               u_threshold, reset_mode);

    // 返回计算结果
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

/*
LIF神经元的反向传播函数。
@params:
    grad_o: at::Tensor 脉冲输出$O^{l}$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}$的梯度
    grad_h: at::Tensor 胞体历史电位$H^{l}$的梯度
    grad_x: at::Tensor 输入电位$X^{l}$的梯度
    grad_u_init: at::Tensor 初始胞体电位$H^{l}(-1)$的梯度
    grad_tau_m: at::Tensor 时间常数$τ_{m}$的梯度
    time_steps: int 总时间步长
    o: at::Tensor 脉冲输出$O^{l}$
    u: at::Tensor 胞体电位$U^{l}$
    h: at::Tensor 胞体历史电位$H^{l}$
    x: at::Tensor 输入电位$X^{l}$
    u_init: at::Tensor 初始胞体电位$H^{l}(-1)$
    tau_m: at::Tensor 时间常数$τ_{m}$
    u_rest: float 静息电位$u_{rest}$
    u_threshold: float 阈电位$u_{th}$
    spiking_mode: int 替代梯度模式
    a: float 参数$a$
    reset_mode: int 重置模式，分为硬重置（0）和软重置（1）两种
*/
__global__ void bp_lif_cuda_kernel(float* grad_o,
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
                                   float u_threshold,
                                   int spiking_mode,
                                   float a,
                                   int reset_mode) {
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
        switch (reset_mode) {
            case RESET_HARD:
                bp_reset_hard(cur_grad_h, grad_u[cur_idx], grad_o[cur_idx],
                              h[cur_idx], u[cur_idx], o[cur_idx], u_rest);
                break;
            case RESET_SOFT:
                bp_reset_soft(cur_grad_h, grad_u[cur_idx], grad_o[cur_idx],
                              h[cur_idx], u[cur_idx], o[cur_idx], u_threshold,
                              u_rest);
                break;
        }
        switch (spiking_mode) {
            case SURROGATE_RECTANGULAR:
                bp_spiking_rectangular(grad_o[cur_idx], grad_u[cur_idx],
                                       o[cur_idx], u[cur_idx], u_threshold, a);
                break;
            case SURROGATE_POLYNOMIAL:
                bp_spiking_polynomial(grad_o[cur_idx], grad_u[cur_idx],
                                      o[cur_idx], u[cur_idx], u_threshold, a);
                break;
            case SURROGATE_SIGMOID:
                bp_spiking_sigmoid(grad_o[cur_idx], grad_u[cur_idx], o[cur_idx],
                                   u[cur_idx], u_threshold, a);
                break;
            case SURROGATE_GAUSSIAN:
                bp_spiking_gaussian(grad_o[cur_idx], grad_u[cur_idx],
                                    o[cur_idx], u[cur_idx], u_threshold, a);
                break;
        }
        bp_response_lif(grad_u[cur_idx], grad_x[cur_idx], cur_grad_h,
                        grad_tau_m[0], u[cur_idx], x[cur_idx], last_h,
                        tau_m_val, u_rest);
        if (t) {
            grad_h[cur_idx - shape] = cur_grad_h;
        } else {
            grad_u_init[idx] = cur_grad_h;
        }
    }
}

void bp_lif_cuda(float* grad_o,
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
                 float u_threshold,
                 int spiking_mode,
                 float a,
                 int reset_mode) {
    cudaError_t err;

    // i = blockIdx.y 为行
    // j = blockIdx.x * blockDim.x + threadIdx.x 为列
    dim3 blocks(DIVUP(DIVUP(shape, 1024), THREADS_PER_BLOCK), 1024);
    dim3 threads(THREADS_PER_BLOCK);

    // 调用CUDA核心开始计算
    bp_lif_cuda_kernel<<<blocks, threads, 0>>>(
        grad_o, grad_u, grad_h, grad_x, grad_u_init, grad_tau_m, time_steps,
        shape, o, u, h, x, u_init, tau_m, u_rest, u_threshold, spiking_mode, a,
        reset_mode);

    // 返回计算结果
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}