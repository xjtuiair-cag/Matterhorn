#ifndef _SOMA_H
#define _SOMA_H

#include <torch/serialize/tensor.h>
#include <vector>

int fp_response_lif(at::Tensor u,
                    at::Tensor x,
                    at::Tensor h,
                    at::Tensor tau_m,
                    float u_rest);

int fp_spiking_heaviside(at::Tensor o, at::Tensor u, float u_threshold);

int fp_reset_hard(at::Tensor h, at::Tensor u, at::Tensor o, float u_rest);

int bp_response_lif(at::Tensor grad_u,
                    at::Tensor grad_x,
                    at::Tensor grad_h,
                    at::Tensor grad_tau_m,
                    at::Tensor u,
                    at::Tensor x,
                    at::Tensor h,
                    at::Tensor tau_m,
                    float u_rest);

int bp_spiking_rectangular(at::Tensor grad_o,
                           at::Tensor grad_u,
                           at::Tensor o,
                           at::Tensor u,
                           float u_threshold);

int bp_reset_hard(at::Tensor grad_h,
                  at::Tensor grad_u,
                  at::Tensor grad_o,
                  at::Tensor h,
                  at::Tensor u,
                  at::Tensor o,
                  float u_rest);

int fp_lif(at::Tensor o,
           at::Tensor u,
           at::Tensor h,
           at::Tensor x,
           int time_steps,
           at::Tensor u_init,
           at::Tensor tau_m,
           float u_rest,
           float u_threshold);

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
           float u_threshold);

#endif