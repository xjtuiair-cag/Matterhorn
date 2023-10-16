#include "soma.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>

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
void cu_fp_lif_heaviside_hard(at::Tensor o,
                              at::Tensor u,
                              at::Tensor h,
                              at::Tensor x,
                              int time_steps,
                              int shape,
                              at::Tensor u_init,
                              at::Tensor tau_m,
                              float u_rest,
                              float u_threshold) {
    float* o_head = o.data_ptr<float>();
    float* u_head = u.data_ptr<float>();
    float* h_head = h.data_ptr<float>();
    const float* x_head = x.data_ptr<float>();
    const float* u_init_head = u_init.data_ptr<float>();
    const float* tau_m_head = tau_m.data_ptr<float>();

    fp_lif_heaviside_hard_cuda(o_head, u_head, h_head, x_head, time_steps,
                               shape, u_init_head, tau_m_head, u_rest,
                               u_threshold);
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
void cu_bp_lif_rectangular_hard(at::Tensor grad_o,
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
                                float u_rest,
                                float u_threshold) {
    float* grad_o_head = grad_o.data_ptr<float>();
    float* grad_u_head = grad_u.data_ptr<float>();
    float* grad_h_head = grad_h.data_ptr<float>();
    float* grad_x_head = grad_x.data_ptr<float>();
    float* grad_u_init_head = grad_u_init.data_ptr<float>();
    float* grad_tau_m_head = grad_tau_m.data_ptr<float>();
    const float* o_head = o.data_ptr<float>();
    const float* u_head = u.data_ptr<float>();
    const float* h_head = h.data_ptr<float>();
    const float* x_head = x.data_ptr<float>();
    const float* u_init_head = u_init.data_ptr<float>();
    const float* tau_m_head = tau_m.data_ptr<float>();

    bp_lif_rectangular_hard_cuda(
        grad_o_head, grad_u_head, grad_h_head, grad_x_head, grad_u_init_head,
        grad_tau_m_head, time_steps, shape, o_head, u_head, h_head, x_head,
        u_init_head, tau_m_head, u_rest, u_threshold);
}