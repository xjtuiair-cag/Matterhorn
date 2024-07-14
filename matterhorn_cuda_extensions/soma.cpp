#ifndef _MATTERHORN_SOMA_CUDA
#define _MATTERHORN_SOMA_CUDA


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <ATen/ATen.h>
#include "base.h"
#include "soma.h"

using namespace std;

/*
LIF神经元的前向传播函数。
Args:
    o (at::Tensor): 脉冲输出$O^{l}$
    u (at::Tensor): 胞体电位$U^{l}$
    h (at::Tensor): 胞体历史电位$H^{l}$
    x (at::Tensor): 输入电位$X^{l}$
    time_steps (int): 总时间步长
    u_init (at::Tensor): 初始胞体电位$H^{l}(-1)$
    tau_m (at::Tensor): 时间常数$τ_{m}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
    u_threshold (at::Tensor): 阈电位$u_{th}$
    reset_mode (int): 重置模式，分为硬重置（0）和软重置（1）两种
*/
void cu_fp_lif(at::Tensor o,
               at::Tensor u,
               at::Tensor h,
               at::Tensor x,
               int time_steps,
               int shape,
               at::Tensor u_init,
               at::Tensor tau_m,
               at::Tensor u_rest,
               at::Tensor u_threshold,
               int firing_mode = FIRING_GAUSSIAN,
               int reset_mode = RESET_HARD) {
    float* o_head = o.data_ptr<float>();
    float* u_head = u.data_ptr<float>();
    float* h_head = h.data_ptr<float>();
    float* x_head = x.data_ptr<float>();
    float* u_init_head = u_init.data_ptr<float>();
    float* tau_m_head = tau_m.data_ptr<float>();
    float* u_threshold_head = u_threshold.data_ptr<float>();
    float* u_rest_head = u_rest.data_ptr<float>();

    fp_lif_cuda(o_head, u_head, h_head, x_head, time_steps, shape, u_init_head,
                tau_m_head, u_rest_head, u_threshold_head, firing_mode, reset_mode);
}

/*
LIF神经元的反向传播函数。
Args:
    grad_o (at::Tensor): 脉冲输出$O^{l}$的梯度
    grad_u (at::Tensor): 胞体电位$U^{l}$的梯度
    grad_h (at::Tensor): 胞体历史电位$H^{l}$的梯度
    grad_x (at::Tensor): 输入电位$X^{l}$的梯度
    grad_u_init (at::Tensor): 初始胞体电位$H^{l}(-1)$的梯度
    grad_tau_m (at::Tensor): 时间常数$τ_{m}$的梯度
    time_steps (int): 总时间步长
    o (at::Tensor): 脉冲输出$O^{l}$
    u (at::Tensor): 胞体电位$U^{l}$
    h (at::Tensor): 胞体历史电位$H^{l}$
    x (at::Tensor): 输入电位$X^{l}$
    u_init (at::Tensor): 初始胞体电位$H^{l}(-1)$
    tau_m (at::Tensor): 时间常数$τ_{m}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
    u_threshold (at::Tensor): 阈电位$u_{th}$
    firing_mode (int): 替代梯度模式
    a (float): 参数$a$
    reset_mode (int): 重置模式，分为硬重置（0）和软重置（1）两种
*/
void cu_bp_lif(at::Tensor grad_o,
               at::Tensor grad_u,
               at::Tensor grad_h,
               at::Tensor grad_x,
               at::Tensor grad_u_init,
               at::Tensor grad_tau_m,
               int time_steps,
               int shape,
               at::Tensor o,
               at::Tensor u,
               at::Tensor h,
               at::Tensor x,
               at::Tensor u_init,
               at::Tensor tau_m,
               at::Tensor u_rest,
               at::Tensor u_threshold,
               int firing_mode = FIRING_GAUSSIAN,
               float a = 4.0,
               int reset_mode = RESET_HARD) {
    float* grad_o_head = grad_o.data_ptr<float>();
    float* grad_u_head = grad_u.data_ptr<float>();
    float* grad_h_head = grad_h.data_ptr<float>();
    float* grad_x_head = grad_x.data_ptr<float>();
    float* grad_u_init_head = grad_u_init.data_ptr<float>();
    float* grad_tau_m_head = grad_tau_m.data_ptr<float>();
    float* o_head = o.data_ptr<float>();
    float* u_head = u.data_ptr<float>();
    float* h_head = h.data_ptr<float>();
    float* x_head = x.data_ptr<float>();
    float* u_init_head = u_init.data_ptr<float>();
    float* tau_m_head = tau_m.data_ptr<float>();
    float* u_threshold_head = u_threshold.data_ptr<float>();
    float* u_rest_head = u_rest.data_ptr<float>();

    bp_lif_cuda(grad_o_head, grad_u_head, grad_h_head, grad_x_head,
                grad_u_init_head, grad_tau_m_head, time_steps, shape, o_head,
                u_head, h_head, x_head, u_init_head, tau_m_head, u_rest_head,
                u_threshold_head, firing_mode, a, reset_mode);
}


#endif