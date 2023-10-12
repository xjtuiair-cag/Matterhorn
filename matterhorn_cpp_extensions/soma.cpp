#include "soma.h"
#include <torch/serialize/tensor.h>
#include <cmath>
#include <vector>

int fp_response_lif(at::Tensor u,
                    at::Tensor x,
                    at::Tensor h,
                    at::Tensor tau_m,
                    float u_rest) {
    float tau_m_val = tau_m.data<float>()[0];
    at::Tensor du = (1.0 / tau_m_val) * (-(h - u_rest) + x);
    u += h + du;
}

int fp_spiking_heaviside(at::Tensor o, at::Tensor u, float u_threshold) {
    o += (u >= u_threshold);
}

int fp_reset_hard(at::Tensor h, at::Tensor u, at::Tensor o, float u_rest) {
    h += u * (1.0 - o) + u_rest * o;
}

int bp_response_lif(at::Tensor grad_u,
                    at::Tensor grad_x,
                    at::Tensor grad_h,
                    at::Tensor grad_tau_m,
                    at::Tensor u,
                    at::Tensor x,
                    at::Tensor h,
                    at::Tensor tau_m,
                    float u_rest) {
    float tau_m_val = tau_m.data<float>()[0];
    grad_x += grad_u * (1.0 / tau_m_val);
    grad_h += grad_u * (1.0 - (1.0 / tau_m_val));
    grad_tau_m += grad_u * (-(1 / tau_m_val / tau_m_val) * (-(h - u_rest) + x));
}

int bp_spiking_rectangular(at::Tensor grad_o,
                           at::Tensor grad_u,
                           at::Tensor o,
                           at::Tensor u,
                           float u_threshold) {
    grad_u = grad_o * ((u >= u_threshold - 1) & (u <= u_threshold + 1));
}

int bp_reset_hard(at::Tensor grad_h,
                  at::Tensor grad_u,
                  at::Tensor grad_o,
                  at::Tensor h,
                  at::Tensor u,
                  at::Tensor o,
                  float u_rest) {
    grad_u += grad_h * 1 - o;
    grad_o += grad_h * (u_rest - u);
}

/*
LIF神经元的前向传播函数
@params:
    o: at::Tensor 脉冲输出o

*/
int fp_lif(at::Tensor o,
           at::Tensor u,
           at::Tensor h,
           at::Tensor x,
           int time_steps,
           at::Tensor u_init,
           at::Tensor tau_m,
           float u_rest,
           float u_threshold) {
    for (int t = 0; t < time_steps; t++) {
        fp_response_lif(u[t], x[t], t ? h[t - 1] : u_init, tau_m, u_rest);
        fp_spiking_heaviside(o[t], u[t], u_threshold);
        fp_reset_hard(h[t], u[t], o[t], u_rest);
    }
}

int bp_lif(at::Tensor grad_o,
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
           float u_threshold) {
    for (int t = time_steps - 1; t >= 0; t--) {
        bp_reset_hard(grad_h[t], grad_u[t], grad_o[t], h[t], u[t], o[t],
                      u_rest);
        bp_spiking_rectangular(grad_o[t], grad_u[t], o[t], u[t], u_threshold);
        bp_response_lif(grad_u[t], grad_x[t], t ? grad_h[t - 1] : grad_u_init,
                        grad_tau_m, u[t], x[t], t ? h[t] : u_init, tau_m,
                        u_rest);
    }
}