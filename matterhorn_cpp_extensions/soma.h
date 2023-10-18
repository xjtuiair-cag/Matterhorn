#ifndef _MATTERHORN_SOMA_H
#define _MATTERHORN_SOMA_H

#include <ATen/ATen.h>
#include <vector>

void fp_response_lif(at::Tensor u,
                     at::Tensor x,
                     at::Tensor h,
                     at::Tensor tau_m,
                     float u_rest);

void bp_response_lif(at::Tensor grad_u,
                     at::Tensor grad_x,
                     at::Tensor grad_h,
                     at::Tensor grad_tau_m,
                     at::Tensor u,
                     at::Tensor x,
                     at::Tensor h,
                     at::Tensor tau_m,
                     float u_rest);

void fp_spiking_heaviside(at::Tensor o, at::Tensor u, float u_threshold);

void bp_spiking_rectangular(at::Tensor grad_o,
                            at::Tensor grad_u,
                            at::Tensor o,
                            at::Tensor u,
                            float u_threshold,
                            float a);

void bp_spiking_polynomial(at::Tensor grad_o,
                           at::Tensor grad_u,
                           at::Tensor o,
                           at::Tensor u,
                           float u_threshold,
                           float a);

void bp_spiking_sigmoid(at::Tensor grad_o,
                        at::Tensor grad_u,
                        at::Tensor o,
                        at::Tensor u,
                        float u_threshold,
                        float a);

void bp_spiking_gaussian(at::Tensor grad_o,
                         at::Tensor grad_u,
                         at::Tensor o,
                         at::Tensor u,
                         float u_threshold,
                         float a);

void fp_reset_hard(at::Tensor h, at::Tensor u, at::Tensor o, float u_rest);

void bp_reset_hard(at::Tensor grad_h,
                   at::Tensor grad_u,
                   at::Tensor grad_o,
                   at::Tensor h,
                   at::Tensor u,
                   at::Tensor o,
                   float u_rest);

void fp_reset_soft(at::Tensor h,
                   at::Tensor u,
                   at::Tensor o,
                   float u_threshold,
                   float u_rest);

void bp_reset_soft(at::Tensor grad_h,
                   at::Tensor grad_u,
                   at::Tensor grad_o,
                   at::Tensor h,
                   at::Tensor u,
                   at::Tensor o,
                   float u_threshold,
                   float u_rest);

void fp_lif(at::Tensor o,
            at::Tensor u,
            at::Tensor h,
            at::Tensor x,
            int time_steps,
            at::Tensor u_init,
            at::Tensor tau_m,
            float u_rest,
            float u_threshold,
            int reset_mode);

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
            int spiking_mode,
            float a,
            int reset_mode);

#endif