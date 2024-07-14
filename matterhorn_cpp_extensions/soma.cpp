#ifndef _MATTERHORN_SOMA
#define _MATTERHORN_SOMA

#include "soma.h"
#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <cmath>
#include <iostream>
#include <vector>
#include "base.h"

using namespace std;

/*
IF神经元反应函数的前向传播函数。
$$U_{i}^{l}(t)=H_{i}^{l}(t-1)+X_{i}^{l}(t)$$
Args:
    u (at::Tensor): 胞体电位$U^{l}(t)$
    x (at::Tensor): 输入电位$X^{l}(t)$
    h (at::Tensor): 胞体历史电位$H^{l}(t-1)$
*/
void fp_response_if(at::Tensor u, at::Tensor x, at::Tensor h) {
    at::Tensor du = x;
    u += h + du;
}

/*
IF神经元反应函数的反向传播函数。
$$\frac{\partial U_{i}^{l}(t)}{\partial H_{i}^{l}(t-1)}=1$$
$$\frac{\partial U_{i}^{l}(t)}{\partial X_{i}^{l}(t)}=1$$
Args:
    grad_u (at::Tensor): 胞体电位$U^{l}(t)$的梯度
    grad_x (at::Tensor): 输入电位$X^{l}(t)$的梯度
    grad_h (at::Tensor): 胞体历史电位$H^{l}(t-1)$的梯度
    u (at::Tensor): 胞体电位$U^{l}(t)$
    x (at::Tensor): 输入电位$X^{l}(t-1)$
    h (at::Tensor): 胞体历史电位$H^{l}(t)$
*/
void bp_response_if(at::Tensor grad_u,
                    at::Tensor grad_x,
                    at::Tensor grad_h,
                    at::Tensor u,
                    at::Tensor x,
                    at::Tensor h) {
    grad_x += grad_u * 1.0f;
    grad_h += grad_u * 1.0f;
}

/*
LIF神经元反应函数的前向传播函数。
$$U_{i}^{l}(t)=H_{i}^{l}(t-1)+\frac{1}{τ_{m}}[-[H_{i}^{l}(t-1)-u_{rest}]+X_{i}^{l}(t)]$$
Args:
    u (at::Tensor): 胞体电位$U^{l}(t)$
    x (at::Tensor): 输入电位$X^{l}(t)$
    h (at::Tensor): 胞体历史电位$H^{l}(t-1)$
    tau_m (at::Tensor): 时间常数$τ_{m}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void fp_response_lif(at::Tensor u,
                     at::Tensor x,
                     at::Tensor h,
                     at::Tensor tau_m,
                     at::Tensor u_rest) {
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
    grad_u (at::Tensor): 胞体电位$U^{l}(t)$的梯度
    grad_x (at::Tensor): 输入电位$X^{l}(t)$的梯度
    grad_h (at::Tensor): 胞体历史电位$H^{l}(t-1)$的梯度
    grad_tau_m (at::Tensor): 时间常数$τ_{m}$的梯度
    u (at::Tensor): 胞体电位$U^{l}(t)$
    x (at::Tensor): 输入电位$X^{l}(t-1)$
    h (at::Tensor): 胞体历史电位$H^{l}(t)$
    tau_m (at::Tensor): 时间常数$τ_{m}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void bp_response_lif(at::Tensor grad_u,
                     at::Tensor grad_x,
                     at::Tensor grad_h,
                     at::Tensor grad_tau_m,
                     at::Tensor u,
                     at::Tensor x,
                     at::Tensor h,
                     at::Tensor tau_m,
                     at::Tensor u_rest) {
    float tau_m_val = tau_m.data<float>()[0];
    grad_x += grad_u * (1.0f / tau_m_val);
    grad_h += grad_u * (1.0f - (1.0f / tau_m_val));
    grad_tau_m +=
        grad_u * (-1.0f / powf(tau_m_val, 2.0f)) * (-(h - u_rest) + x);
}

/*
QIF神经元反应函数的前向传播函数。
$$U_{i}^{l}(t)=H_{i}^{l}(t-1)+\frac{1}{τ_{m}}[a_{0}(H_{i}^{l}(t-1)-u_{rest})(H_{i}^{l}(t-1)-u_{c})+X_{i}^{l}(t)]$$
Args:
    u (at::Tensor): 胞体电位$U^{l}(t)$
    x (at::Tensor): 输入电位$X^{l}(t)$
    h (at::Tensor): 胞体历史电位$H^{l}(t-1)$
    tau_m (at::Tensor): 时间常数$τ_{m}$
    u_c (at::Tensor): 参数$u_{c}$
    a_0 (at::Tensor): 参数$a_{0}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void fp_response_qif(at::Tensor u,
                     at::Tensor x,
                     at::Tensor h,
                     at::Tensor tau_m,
                     at::Tensor u_c,
                     at::Tensor a_0,
                     at::Tensor u_rest) {
    float tau_m_val = tau_m.data<float>()[0];
    float u_c_val = u_c.data<float>()[0];
    float a_0_val = a_0.data<float>()[0];
    at::Tensor du =
        (1.0f / tau_m_val) * (a_0_val * (h - u_rest) * (h - u_c_val) + x);
    u += h + du;
}

/*
QIF神经元反应函数的反向传播函数。
$$\frac{\partial U_{i}^{l}(t)}{\partial
H_{i}^{l}(t-1)}=1+\frac{a_{0}}{τ_{m}}[2H_{i}^{l}(t-1)-u_{rest}-u_{c}]$$
$$\frac{\partial U_{i}^{l}(t)}{\partial X_{i}^{l}(t)}=\frac{1}{τ_{m}}$$
$$\frac{\partial U_{i}^{l}(t)}{\partial
τ_{m}}=-\frac{1}{τ_{m}^{2}}[a_{0}(H_{i}^{l}(t-1)-u_{rest})(H_{i}^{l}(t-1)-u_{c})+X_{i}^{l}(t)]$$
$$\frac{\partial U_{i}^{l}(t)}{\partial
u_{c}}=-\frac{a_{0}}{τ_{m}}(H_{i}^{l}(t-1)-u_{rest})$$
$$\frac{\partial U_{i}^{l}(t)}{\partial
a_{0}}=\frac{1}{τ_{m}}(H_{i}^{l}(t-1)-u_{rest})(H_{i}^{l}(t-1)-u_{c})$$
Args:
    grad_u (at::Tensor): 胞体电位$U^{l}(t)$的梯度
    grad_x (at::Tensor): 输入电位$X^{l}(t)$的梯度
    grad_h (at::Tensor): 胞体历史电位$H^{l}(t-1)$的梯度
    grad_tau_m (at::Tensor): 时间常数$τ_{m}$的梯度
    grad_u_c (at::Tensor): 参数$u_{c}$的梯度
    grad_a_0 (at::Tensor): 参数$a_{0}$的梯度
    u (at::Tensor): 胞体电位$U^{l}(t)$
    x (at::Tensor): 输入电位$X^{l}(t-1)$
    h (at::Tensor): 胞体历史电位$H^{l}(t)$
    tau_m (at::Tensor): 时间常数$τ_{m}$
    u_c (at::Tensor): 参数$u_{c}$
    a_0 (at::Tensor): 参数$a_{0}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void bp_response_qif(at::Tensor grad_u,
                     at::Tensor grad_x,
                     at::Tensor grad_h,
                     at::Tensor grad_tau_m,
                     at::Tensor grad_u_c,
                     at::Tensor grad_a_0,
                     at::Tensor u,
                     at::Tensor x,
                     at::Tensor h,
                     at::Tensor tau_m,
                     at::Tensor u_c,
                     at::Tensor a_0,
                     at::Tensor u_rest) {
    float tau_m_val = tau_m.data<float>()[0];
    float u_c_val = u_c.data<float>()[0];
    float a_0_val = a_0.data<float>()[0];
    grad_x += grad_u * (1.0f / tau_m_val);
    grad_h +=
        grad_u * (1.0f + (a_0_val / tau_m_val) * (2.0f * h - u_rest - u_c_val));
    grad_tau_m += grad_u * (-1.0f / powf(tau_m_val, 2.0f)) *
                  (a_0_val * (h - u_rest) * (h - u_c_val) + x);
    grad_u_c += grad_u * -(a_0_val / tau_m_val) * (h - u_rest);
    grad_a_0 += grad_u * (1.0f / tau_m_val) * (h - u_rest) * (h - u_c_val);
}

/*
ExpIF神经元反应函数的前向传播函数。
$$U_{i}^{l}(t)=H_{i}^{l}(t-1)+\frac{1}{τ_{m}}[-(H_{i}^{l}(t-1)-u_{rest})+Δ_{T}e^{\frac{H_{i}^{l}(t-1)-u_{T}}{Δ_{T}}}+X_{i}^{l}(t)]$$
Args:
    u (at::Tensor): 胞体电位$U^{l}(t)$
    x (at::Tensor): 输入电位$X^{l}(t)$
    h (at::Tensor): 胞体历史电位$H^{l}(t-1)$
    tau_m (at::Tensor): 时间常数$τ_{m}$
    u_t (at::Tensor): 参数$u_{t}$
    delta_t (at::Tensor): 参数$Δ_{T}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void fp_response_expif(at::Tensor u,
                       at::Tensor x,
                       at::Tensor h,
                       at::Tensor tau_m,
                       at::Tensor u_t,
                       at::Tensor delta_t,
                       at::Tensor u_rest) {
    float tau_m_val = tau_m.data<float>()[0];
    float u_t_val = u_t.data<float>()[0];
    float delta_t_val = delta_t.data<float>()[0];
    at::Tensor du = (1.0f / tau_m_val) *
                    (-(h - u_rest) +
                     delta_t_val * at::exp((h - u_t_val) / delta_t_val) + x);
    u += h + du;
}

/*
ExpIF神经元反应函数的反向传播函数。
$$\frac{\partial U_{i}^{l}(t)}{\partial
H_{i}^{l}(t-1)}=1-\frac{1}{τ_{m}}+e^{\frac{H_{i}^{l}(t-1)-u_{T}}{Δ_{T}}}$$
$$\frac{\partial U_{i}^{l}(t)}{\partial X_{i}^{l}(t)}=\frac{1}{τ_{m}}$$
$$\frac{\partial U_{i}^{l}(t)}{\partial
τ_{m}}=-\frac{1}{τ_{m}^{2}}[-(H_{i}^{l}(t-1)-u_{rest})+Δ_{T}e^{\frac{H_{i}^{l}(t-1)-u_{T}}{Δ_{T}}}+X_{i}^{l}(t)]$$
$$\frac{\partial U_{i}^{l}(t)}{\partial
u_{t}}=-\frac{1}{τ_{m}}e^{\frac{H_{i}^{l}(t-1)-u_{T}}{Δ_{T}}}$$
$$\frac{\partial U_{i}^{l}(t)}{\partial
Δ_{T}}=\frac{1}{τ_{m}}(1-\frac{1}{Δ_{T}})e^{\frac{H_{i}^{l}(t-1)-u_{T}}{Δ_{T}}}$$
Args:
    grad_u (at::Tensor): 胞体电位$U^{l}(t)$的梯度
    grad_x (at::Tensor): 输入电位$X^{l}(t)$的梯度
    grad_h (at::Tensor): 胞体历史电位$H^{l}(t-1)$的梯度
    grad_tau_m (at::Tensor): 时间常数$τ_{m}$的梯度
    grad_u_t (at::Tensor): 参数$u_{t}$的梯度
    grad_delta_t (at::Tensor): 参数$Δ_{T}$的梯度
    u (at::Tensor): 胞体电位$U^{l}(t)$
    x (at::Tensor): 输入电位$X^{l}(t-1)$
    h (at::Tensor): 胞体历史电位$H^{l}(t)$
    tau_m (at::Tensor): 时间常数$τ_{m}$
    u_t (at::Tensor): 参数$u_{t}$
    delta_t (at::Tensor): 参数$Δ_{T}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void bp_response_expif(at::Tensor grad_u,
                       at::Tensor grad_x,
                       at::Tensor grad_h,
                       at::Tensor grad_tau_m,
                       at::Tensor grad_u_t,
                       at::Tensor grad_delta_t,
                       at::Tensor u,
                       at::Tensor x,
                       at::Tensor h,
                       at::Tensor tau_m,
                       at::Tensor u_t,
                       at::Tensor delta_t,
                       at::Tensor u_rest) {
    float tau_m_val = tau_m.data<float>()[0];
    float u_t_val = u_t.data<float>()[0];
    float delta_t_val = delta_t.data<float>()[0];
    grad_x += grad_u * (1.0f / tau_m_val);
    grad_h += grad_u * (1.0f - (1.0f / tau_m_val) +
                        at::exp((h - u_t_val) / delta_t_val));
    grad_tau_m += grad_u * (-1.0f / powf(tau_m_val, 2.0f)) *
                  (-(h - u_rest) +
                   delta_t_val * at::exp((h - u_t_val) / delta_t_val) + x);
    grad_u_t +=
        grad_u * (-1.0f / tau_m_val) * at::exp((h - u_t_val) / delta_t_val);
    grad_delta_t += grad_u * (1.0f / tau_m_val) * (1.0f - 1.0f / delta_t_val) *
                    at::exp((h - u_t_val) / delta_t_val);
}

/*
Heaviside脉冲函数前向传播函数。
$$O_{i}^{l}(t)=u[U_{i}^{l}(t)]$$
Args:
    o (at::Tensor): 脉冲输出$O^{l}(t)$
    u (at::Tensor): 胞体电位$U^{l}(t)$
    u_threshold (at::Tensor): 阈电位$u_{th}$
*/
void fp_spiking_heaviside(at::Tensor o, at::Tensor u, at::Tensor u_threshold) {
    o += at::ge(u, u_threshold);
}

/*
多值向下取整前向传播函数。
$$O_{i}^{l}(t)=\floor{U_{i}^{l}(t)}$$
Args:
    o (at::Tensor): 脉冲输出$O^{l}(t)$
    u (at::Tensor): 胞体电位$U^{l}(t)$
    u_threshold (at::Tensor): 阈电位$u_{th}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void fp_spiking_floor(at::Tensor o,
                      at::Tensor u,
                      at::Tensor u_threshold,
                      at::Tensor u_rest) {
    o += at::floor((u - u_rest) / (u_threshold - u_rest));
}

/*
多值向上取整前向传播函数。
$$O_{i}^{l}(t)=\ceil{U_{i}^{l}(t)}$$
Args:
    o (at::Tensor): 脉冲输出$O^{l}(t)$
    u (at::Tensor): 胞体电位$U^{l}(t)$
    u_threshold (at::Tensor): 阈电位$u_{th}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void fp_spiking_ceil(at::Tensor o,
                     at::Tensor u,
                     at::Tensor u_threshold,
                     at::Tensor u_rest) {
    o += at::ceil((u - u_rest) / (u_threshold - u_rest));
}

/*
多值四舍五入前向传播函数。
$$O_{i}^{l}(t)=round(U_{i}^{l}(t))$$
Args:
    o (at::Tensor): 脉冲输出$O^{l}(t)$
    u (at::Tensor): 胞体电位$U^{l}(t)$
    u_threshold (at::Tensor): 阈电位$u_{th}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void fp_spiking_round(at::Tensor o,
                      at::Tensor u,
                      at::Tensor u_threshold,
                      at::Tensor u_rest) {
    o += at::round((u - u_rest) / (u_threshold - u_rest));
}

/*
矩形窗反向传播函数。
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=u'$$
Args:
    grad_o (at::Tensor): 脉冲输出$O^{l}(t)$的梯度
    grad_u (at::Tensor): 胞体电位$U^{l}(t)$的梯度
    o (at::Tensor): 脉冲输出$O^{l}(t)$
    u (at::Tensor): 胞体电位$U^{l}(t)$
    u_threshold (at::Tensor): 阈电位$u_{th}$
    a (float): 参数$a$
*/
void bp_spiking_rectangular(at::Tensor grad_o,
                            at::Tensor grad_u,
                            at::Tensor o,
                            at::Tensor u,
                            at::Tensor u_threshold,
                            float a = 2.0f) {
    at::Tensor ax = u - u_threshold;
    grad_u += grad_o * (1.0f / a) * at::lt(at::abs(ax), a / 2.0f);
}

/*
多项式反向传播函数。
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=u'$$
Args:
    grad_o (at::Tensor): 脉冲输出$O^{l}(t)$的梯度
    grad_u (at::Tensor): 胞体电位$U^{l}(t)$的梯度
    o (at::Tensor): 脉冲输出$O^{l}(t)$
    u (at::Tensor): 胞体电位$U^{l}(t)$
    u_threshold (at::Tensor): 阈电位$u_{th}$
    a (float): 参数$a$
*/
void bp_spiking_polynomial(at::Tensor grad_o,
                           at::Tensor grad_u,
                           at::Tensor o,
                           at::Tensor u,
                           at::Tensor u_threshold,
                           float a = 1.0f) {
    at::Tensor ax = at::abs(u - u_threshold);
    grad_u += grad_o * (sqrtf(a) / 2.0f - a / 4.0f * ax) *
              at::sign(2.0f / sqrtf(a) - ax) * at::lt(ax, 2.0f / sqrtf(a));
}

/*
Sigmoid导数的反向传播函数。
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=u'$$
Args:
    grad_o (at::Tensor): 脉冲输出$O^{l}(t)$的梯度
    grad_u (at::Tensor): 胞体电位$U^{l}(t)$的梯度
    o (at::Tensor): 脉冲输出$O^{l}(t)$
    u (at::Tensor): 胞体电位$U^{l}(t)$
    u_threshold (at::Tensor): 阈电位$u_{th}$
    a (float): 参数$a$
*/
void bp_spiking_sigmoid(at::Tensor grad_o,
                        at::Tensor grad_u,
                        at::Tensor o,
                        at::Tensor u,
                        at::Tensor u_threshold,
                        float a = 1.0f) {
    at::Tensor ax = u - u_threshold;
    at::Tensor ex = at::exp(-ax / a);
    grad_u += grad_o * (1.0f / a) * ex / at::pow(1.0f + ex, 2.0f);
}

/*
高斯反向传播函数。
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=u'$$
Args:
    grad_o (at::Tensor): 脉冲输出$O^{l}(t)$的梯度
    grad_u (at::Tensor): 胞体电位$U^{l}(t)$的梯度
    o (at::Tensor): 脉冲输出$O^{l}(t)$
    u (at::Tensor): 胞体电位$U^{l}(t)$
    u_threshold (at::Tensor): 阈电位$u_{th}$
    a (float): 参数$a$
*/
void bp_spiking_gaussian(at::Tensor grad_o,
                         at::Tensor grad_u,
                         at::Tensor o,
                         at::Tensor u,
                         at::Tensor u_threshold,
                         float a = 1.0f) {
    at::Tensor ax = u - u_threshold;
    grad_u += grad_o * 1.0f / sqrtf(2.0f * M_PI * a) *
              at::exp(-at::pow(ax, 2.0f) / (2.0f * a));
}

/*
多值反向传播函数。
$$\frac{\partial O_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=U_{i}^{l}(t)$$
Args:
    grad_o (at::Tensor): 脉冲输出$O^{l}(t)$的梯度
    grad_u (at::Tensor): 胞体电位$U^{l}(t)$的梯度
    o (at::Tensor): 脉冲输出$O^{l}(t)$
    u (at::Tensor): 胞体电位$U^{l}(t)$
    u_threshold (at::Tensor): 阈电位$u_{th}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void bp_spiking_multi(at::Tensor grad_o,
                      at::Tensor grad_u,
                      at::Tensor o,
                      at::Tensor u,
                      at::Tensor u_threshold,
                      at::Tensor u_rest) {
    grad_u += (u - u_rest) / (u_threshold - u_rest);
}

/*
硬重置前向传播函数。
$$H_{i}^{l}(t)=U_{i}^{l}(t)[1-O_{i}^{l}(t)]+u_{rest}O_{i}^{l}(t)$$
Args:
    h (at::Tensor): 胞体历史电位$H^{l}(t)$
    u (at::Tensor): 胞体电位$U^{l}(t)$
    o (at::Tensor): 脉冲输出$O^{l}(t)$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void fp_reset_hard(at::Tensor h,
                   at::Tensor u,
                   at::Tensor o,
                   at::Tensor u_rest) {
    h += u * (1.0f - o) + u_rest * o;
}

/*
硬重置反向传播函数。
$$\frac{\partial H_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=1-O_{i}^{l}(t)$$
$$\frac{\partial H_{i}^{l}(t)}{\partial O_{i}^{l}(t)}=-U_{i}^{l}(t)+u_{rest}$$
Args:
    grad_h (at::Tensor): 胞体历史电位$H^{l}(t)$的梯度
    grad_u (at::Tensor): 胞体电位$U^{l}(t)$的梯度
    grad_o (at::Tensor): 脉冲输出$O^{l}(t)$的梯度
    h (at::Tensor): 胞体历史电位$H^{l}(t)$
    u (at::Tensor): 胞体电位$U^{l}(t)$
    o (at::Tensor): 脉冲输出$O^{l}(t)$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void bp_reset_hard(at::Tensor grad_h,
                   at::Tensor grad_u,
                   at::Tensor grad_o,
                   at::Tensor h,
                   at::Tensor u,
                   at::Tensor o,
                   at::Tensor u_rest) {
    grad_u += grad_h * (1.0f - o);
    grad_o += grad_h * (u_rest - u);
}

/*
软重置前向传播函数。
$$H_{i}^{l}(t)=U_{i}^{l}(t)[1-O_{i}^{l}(t)]+u_{rest}O_{i}^{l}(t)$$
Args:
    h (at::Tensor): 胞体历史电位$H^{l}(t)$
    u (at::Tensor): 胞体电位$U^{l}(t)$
    o (at::Tensor): 脉冲输出$O^{l}(t)$
    u_threshold (at::Tensor): 阈电位$u_{th}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void fp_reset_soft(at::Tensor h,
                   at::Tensor u,
                   at::Tensor o,
                   at::Tensor u_threshold,
                   at::Tensor u_rest) {
    h += u - (u_threshold - u_rest) * o;
}

/*
软重置反向传播函数。
$$\frac{\partial H_{i}^{l}(t)}{\partial U_{i}^{l}(t)}=1-O_{i}^{l}(t)$$
$$\frac{\partial H_{i}^{l}(t)}{\partial O_{i}^{l}(t)}=-U_{i}^{l}(t)+u_{rest}$$
Args:
    grad_h (at::Tensor): 胞体历史电位$H^{l}(t)$的梯度
    grad_u (at::Tensor): 胞体电位$U^{l}(t)$的梯度
    grad_o (at::Tensor): 脉冲输出$O^{l}(t)$的梯度
    h (at::Tensor): 胞体历史电位$H^{l}(t)$
    u (at::Tensor): 胞体电位$U^{l}(t)$
    o (at::Tensor): 脉冲输出$O^{l}(t)$
    u_threshold (at::Tensor): 阈电位$u_{th}$
    u_rest (at::Tensor): 静息电位$u_{rest}$
*/
void bp_reset_soft(at::Tensor grad_h,
                   at::Tensor grad_u,
                   at::Tensor grad_o,
                   at::Tensor h,
                   at::Tensor u,
                   at::Tensor o,
                   at::Tensor u_threshold,
                   at::Tensor u_rest) {
    grad_u += grad_h * 1.0f;
    grad_o += grad_h * -1.0f * (u_threshold - u_rest);
}

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
void fp_lif(at::Tensor o,
            at::Tensor u,
            at::Tensor h,
            at::Tensor x,
            int time_steps,
            at::Tensor u_init,
            at::Tensor tau_m,
            at::Tensor u_rest,
            at::Tensor u_threshold,
            int firing_mode = FIRING_GAUSSIAN,
            int reset_mode = RESET_HARD) {
    for (int t = 0; t < time_steps; t++) {
        fp_response_lif(u[t], x[t], t ? h[t - 1] : u_init, tau_m, u_rest);
        switch (firing_mode) {
            case FIRING_RECTANGULAR:
            case FIRING_POLYNOMIAL:
            case FIRING_SIGMOID:
            case FIRING_GAUSSIAN:
                fp_spiking_heaviside(o[t], u[t], u_threshold);
                break;
            case FIRING_FLOOR:
                fp_spiking_floor(o[t], u[t], u_threshold, u_rest);
                break;
            case FIRING_CEIL:
                fp_spiking_ceil(o[t], u[t], u_threshold, u_rest);
                break;
            case FIRING_ROUND:
                fp_spiking_round(o[t], u[t], u_threshold, u_rest);
                break;
        }
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
            at::Tensor u_rest,
            at::Tensor u_threshold,
            int firing_mode = FIRING_GAUSSIAN,
            float a = 4.0f,
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
        switch (firing_mode) {
            case FIRING_RECTANGULAR:
                bp_spiking_rectangular(grad_o[t], grad_u[t], o[t], u[t],
                                       u_threshold, a);
                break;
            case FIRING_POLYNOMIAL:
                bp_spiking_polynomial(grad_o[t], grad_u[t], o[t], u[t],
                                      u_threshold, a);
                break;
            case FIRING_SIGMOID:
                bp_spiking_sigmoid(grad_o[t], grad_u[t], o[t], u[t],
                                   u_threshold, a);
                break;
            case FIRING_GAUSSIAN:
                bp_spiking_gaussian(grad_o[t], grad_u[t], o[t], u[t],
                                    u_threshold, a);
                break;
            case FIRING_FLOOR:
            case FIRING_CEIL:
            case FIRING_ROUND:
                bp_spiking_multi(grad_o[t], grad_u[t], o[t], u[t], u_threshold,
                                 u_rest);
                break;
        }
        bp_response_lif(grad_u[t], grad_x[t], t ? grad_h[t - 1] : grad_u_init,
                        grad_tau_m, u[t], x[t], t ? h[t - 1] : u_init, tau_m,
                        u_rest);
    }
}

#endif