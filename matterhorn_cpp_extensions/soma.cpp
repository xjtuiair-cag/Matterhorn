#include "base.h"
#include "soma.h"
#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

/*
LIF神经元反应函数的前向传播函数。
$$U_{i}^{l}(t)=H_{i}^{l}(t-1)+\frac{1}{τ_{m}}[-[H_{i}^{l}(t-1)-u_{rest}]+X_{i}^{l}(t)]$$
Args:
    u: at::Tensor 胞体电位$U^{l}(t)$
    x: at::Tensor 输入电位$X^{l}(t)$
    h: at::Tensor 胞体历史电位$H^{l}(t-1)$
    tau_m: at::Tensor 时间常数$τ_{m}$
    u_rest (float): 静息电位$u_{rest}$
*/
void fp_response_lif(at::Tensor u,
                     at::Tensor x,
                     at::Tensor h,
                     at::Tensor tau_m,
                     float u_rest) {
    float tau_m_val = tau_m.data<float>()[0];
    at::Tensor du = (1.0f / tau_m_val) * (-(h - u_rest) + x);
    u += h + du;
}

/*
LIF神经元反应函数的反向传播函数。
$$\frac{\partial U_{i}^{l}(t)}{\partial H_{i}^{l}(t-1)}=1-\frac{1}{τ_{m}}$$
$$\frac{\partial U_{i}^{l}(t)}{\partial X_{i}^{l}(t)}=\frac{1}{τ_{m}}$$
$$\frac{\partial U_{i}^{l}(t)}{\partial
τ_{m}}=-\frac{1}{τ_{m}^{2}}[-[H_{i}^{l}(t-1)-u_{rest}]+X_{i}^{l}(t)]$$
Args:
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    grad_x: at::Tensor 输入电位$X^{l}(t)$的梯度
    grad_h: at::Tensor 胞体历史电位$H^{l}(t-1)$的梯度
    grad_tau_m: at::Tensor 时间常数$τ_{m}$的梯度
    u: at::Tensor 胞体电位$U^{l}(t)$
    h: at::Tensor 胞体历史电位$H^{l}(t)$
    x: at::Tensor 输入电位$X^{l}(t-1)$
    tau_m: at::Tensor 时间常数$τ_{m}$
    u_rest (float): 静息电位$u_{rest}$
*/
void bp_response_lif(at::Tensor grad_u,
                     at::Tensor grad_x,
                     at::Tensor grad_h,
                     at::Tensor grad_tau_m,
                     at::Tensor u,
                     at::Tensor x,
                     at::Tensor h,
                     at::Tensor tau_m,
                     float u_rest) {
    float tau_m_val = tau_m.data<float>()[0];
    grad_x += grad_u * (1.0f / tau_m_val);
    grad_h += grad_u * (1.0f - (1.0f / tau_m_val));
    grad_tau_m += grad_u * (-1.0f / powf(tau_m_val, 2.0f)) * (-(h - u_rest) + x);
}

/*
Heaviside脉冲函数前向传播函数。
$$O_{i}^{l}(t)=u[U_{i}^{l}(t)]$$
Args:
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    u_threshold (float): 阈电位$u_{th}$
*/
void fp_spiking_heaviside(at::Tensor o, at::Tensor u, float u_threshold) {
    o += at::ge(u, u_threshold);
}

/*
矩形窗反向传播函数。
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=u'$$
Args:
    grad_o: at::Tensor 脉冲输出$O^{l}(t)$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    u_threshold (float): 阈电位$u_{th}$
    a (float): 参数$a$
*/
void bp_spiking_rectangular(at::Tensor grad_o,
                            at::Tensor grad_u,
                            at::Tensor o,
                            at::Tensor u,
                            float u_threshold,
                            float a = 2.0f) {
    at::Tensor ax = u - u_threshold;
    grad_u += grad_o * (1.0f / a) * at::lt(at::abs(ax), a / 2.0f);
}

/*
多项式反向传播函数。
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=u'$$
Args:
    grad_o: at::Tensor 脉冲输出$O^{l}(t)$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    u_threshold (float): 阈电位$u_{th}$
    a (float): 参数$a$
*/
void bp_spiking_polynomial(at::Tensor grad_o,
                           at::Tensor grad_u,
                           at::Tensor o,
                           at::Tensor u,
                           float u_threshold,
                           float a = 1.0f) {
    at::Tensor ax = u - u_threshold;
    grad_u += grad_o * (sqrtf(a) / 2.0f - a / 4.0f * ax) *
              at::sign(2.0f / sqrtf(a) - ax);
}

/*
Sigmoid导数的反向传播函数。
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=u'$$
Args:
    grad_o: at::Tensor 脉冲输出$O^{l}(t)$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    u_threshold (float): 阈电位$u_{th}$
    a (float): 参数$a$
*/
void bp_spiking_sigmoid(at::Tensor grad_o,
                        at::Tensor grad_u,
                        at::Tensor o,
                        at::Tensor u,
                        float u_threshold,
                        float a = 1.0f) {
    at::Tensor ax = u - u_threshold;
    at::Tensor ex = at::exp(-ax / a);
    grad_u += grad_o * (1.0f / a) * ex / at::pow(1.0f + ex, 2.0f);
}

/*
高斯反向传播函数。
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=u'$$
Args:
    grad_o: at::Tensor 脉冲输出$O^{l}(t)$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    u_threshold (float): 阈电位$u_{th}$
    a (float): 参数$a$
*/
void bp_spiking_gaussian(at::Tensor grad_o,
                         at::Tensor grad_u,
                         at::Tensor o,
                         at::Tensor u,
                         float u_threshold,
                         float a = 1.0f) {
    at::Tensor ax = u - u_threshold;
    grad_u += grad_o * 1.0f / sqrtf(2.0f * M_PI * a) *
              at::exp(-at::pow(ax, 2.0f) / (2.0f * a));
}

/*
硬重置前向传播函数。
$$H_{i}^{l}(t)=U_{i}^{l}(t)[1-O_{i}^{l}(t)]+u_{rest}O_{i}^{l}(t)$$
Args:
    h: at::Tensor 胞体历史电位$H^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u_rest (float): 静息电位$u_{rest}$
*/
void fp_reset_hard(at::Tensor h, at::Tensor u, at::Tensor o, float u_rest) {
    h += u * (1.0f - o) + u_rest * o;
}

/*
硬重置反向传播函数。
$$\frac{\partial H_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=1-O_{i}^{l}(t)$$
$$\frac{\partial H_{i}^{l}(t)}{\partial O_{i}^{l}(t)}=-U_{i}^{l}(t)+u_{rest}$$
Args:
    grad_h: at::Tensor 胞体历史电位$H^{l}(t)$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    grad_o: at::Tensor 脉冲输出$O^{l}(t)$的梯度
    h: at::Tensor 胞体历史电位$H^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u_rest (float): 静息电位$u_{rest}$
*/
void bp_reset_hard(at::Tensor grad_h,
                   at::Tensor grad_u,
                   at::Tensor grad_o,
                   at::Tensor h,
                   at::Tensor u,
                   at::Tensor o,
                   float u_rest) {
    grad_u += grad_h * (1.0f - o);
    grad_o += grad_h * (u_rest - u);
}

/*
软重置前向传播函数。
$$H_{i}^{l}(t)=U_{i}^{l}(t)[1-O_{i}^{l}(t)]+u_{rest}O_{i}^{l}(t)$$
Args:
    h: at::Tensor 胞体历史电位$H^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u_threshold (float): 阈电位$u_{th}$
    u_rest (float): 静息电位$u_{rest}$
*/
void fp_reset_soft(at::Tensor h,
                   at::Tensor u,
                   at::Tensor o,
                   float u_threshold,
                   float u_rest) {
    h += u - (u_threshold - u_rest) * o;
}

/*
软重置反向传播函数。
$$\frac{\partial H_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=1-O_{i}^{l}(t)$$
$$\frac{\partial H_{i}^{l}(t)}{\partial O_{i}^{l}(t)}=-U_{i}^{l}(t)+u_{rest}$$
Args:
    grad_h: at::Tensor 胞体历史电位$H^{l}(t)$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}(t)$的梯度
    grad_o: at::Tensor 脉冲输出$O^{l}(t)$的梯度
    h: at::Tensor 胞体历史电位$H^{l}(t)$
    u: at::Tensor 胞体电位$U^{l}(t)$
    o: at::Tensor 脉冲输出$O^{l}(t)$
    u_threshold (float): 阈电位$u_{th}$
    u_rest (float): 静息电位$u_{rest}$
*/
void bp_reset_soft(at::Tensor grad_h,
                   at::Tensor grad_u,
                   at::Tensor grad_o,
                   at::Tensor h,
                   at::Tensor u,
                   at::Tensor o,
                   float u_threshold,
                   float u_rest) {
    grad_u += grad_h * 1.0f;
    grad_o += grad_h * -1.0f * (u_threshold - u_rest);
}

/*
LIF神经元的前向传播函数。
Args:
    o: at::Tensor 脉冲输出$O^{l}$
    u: at::Tensor 胞体电位$U^{l}$
    h: at::Tensor 胞体历史电位$H^{l}$
    x: at::Tensor 输入电位$X^{l}$
    time_steps (int): 总时间步长
    u_init: at::Tensor 初始胞体电位$H^{l}(-1)$
    tau_m: at::Tensor 时间常数$τ_{m}$
    u_rest (float): 静息电位$u_{rest}$
    u_threshold (float): 阈电位$u_{th}$
    reset_mode (int): 重置模式，分为硬重置（0）和软重置（1）两种
*/
void fp_lif(at::Tensor o,
            at::Tensor u,
            at::Tensor h,
            at::Tensor x,
            int time_steps,
            at::Tensor u_init,
            at::Tensor tau_m,
            float u_rest,
            float u_threshold,
            int reset_mode = RESET_HARD) {
    for (int t = 0; t < time_steps; t++) {
        fp_response_lif(u[t], x[t], t ? h[t - 1] : u_init, tau_m, u_rest);
        fp_spiking_heaviside(o[t], u[t], u_threshold);
        switch (reset_mode) {
            case RESET_HARD:
                fp_reset_hard(h[t], u[t], o[t], u_rest);
                break;
            case RESET_SOFT:
                fp_reset_soft(h[t], u[t], o[t], u_threshold, u_rest);
                break;
        }
    }
}

/*
LIF神经元的反向传播函数。
Args:
    grad_o: at::Tensor 脉冲输出$O^{l}$的梯度
    grad_u: at::Tensor 胞体电位$U^{l}$的梯度
    grad_h: at::Tensor 胞体历史电位$H^{l}$的梯度
    grad_x: at::Tensor 输入电位$X^{l}$的梯度
    grad_u_init: at::Tensor 初始胞体电位$H^{l}(-1)$的梯度
    grad_tau_m: at::Tensor 时间常数$τ_{m}$的梯度
    time_steps (int): 总时间步长
    o: at::Tensor 脉冲输出$O^{l}$
    u: at::Tensor 胞体电位$U^{l}$
    h: at::Tensor 胞体历史电位$H^{l}$
    x: at::Tensor 输入电位$X^{l}$
    u_init: at::Tensor 初始胞体电位$H^{l}(-1)$
    tau_m: at::Tensor 时间常数$τ_{m}$
    u_rest (float): 静息电位$u_{rest}$
    u_threshold (float): 阈电位$u_{th}$
    spiking_mode (int): 替代梯度模式
    a (float): 参数$a$
    reset_mode (int): 重置模式，分为硬重置（0）和软重置（1）两种
*/
void bp_lif(at::Tensor grad_o,
            at::Tensor grad_u,
            at::Tensor grad_h,
            at::Tensor grad_x,
            at::Tensor grad_u_init,
            at::Tensor grad_tau_m,
            int time_steps,
            at::Tensor o,
            at::Tensor u,
            at::Tensor h,
            at::Tensor x,
            at::Tensor u_init,
            at::Tensor tau_m,
            float u_rest,
            float u_threshold,
            int spiking_mode = SURROGATE_RECTANGULAR,
            float a = 2.0f,
            int reset_mode = RESET_HARD) {
    for (int t = time_steps - 1; t >= 0; t--) {
        switch (reset_mode) {
            case RESET_HARD:
                bp_reset_hard(grad_h[t], grad_u[t], grad_o[t], h[t], u[t], o[t],
                              u_rest);
                break;
            case RESET_SOFT:
                bp_reset_soft(grad_h[t], grad_u[t], grad_o[t], h[t], u[t], o[t],
                              u_threshold, u_rest);
                break;
        }
        switch (spiking_mode) {
            case SURROGATE_RECTANGULAR:
                bp_spiking_rectangular(grad_o[t], grad_u[t], o[t], u[t],
                                       u_threshold, a);
                break;
            case SURROGATE_POLYNOMIAL:
                bp_spiking_polynomial(grad_o[t], grad_u[t], o[t], u[t],
                                      u_threshold, a);
                break;
            case SURROGATE_SIGMOID:
                bp_spiking_sigmoid(grad_o[t], grad_u[t], o[t], u[t],
                                   u_threshold, a);
                break;
            case SURROGATE_GAUSSIAN:
                bp_spiking_gaussian(grad_o[t], grad_u[t], o[t], u[t],
                                    u_threshold, a);
                break;
        }
        bp_response_lif(grad_u[t], grad_x[t], t ? grad_h[t - 1] : grad_u_init,
                        grad_tau_m, u[t], x[t], t ? h[t - 1] : u_init, tau_m,
                        u_rest);
    }
}