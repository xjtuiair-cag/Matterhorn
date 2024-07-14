#ifndef _MATTERHORN_SOMA_H
#define _MATTERHORN_SOMA_H

#include <ATen/ATen.h>
#include <vector>

void fp_response_if(at::Tensor u, at::Tensor x, at::Tensor h);

void bp_response_if(at::Tensor grad_u,
                    at::Tensor grad_x,
                    at::Tensor grad_h,
                    at::Tensor u,
                    at::Tensor x,
                    at::Tensor h);

void fp_response_lif(at::Tensor u,
                     at::Tensor x,
                     at::Tensor h,
                     at::Tensor tau_m,
                     at::Tensor u_rest);

void bp_response_lif(at::Tensor grad_u,
                     at::Tensor grad_x,
                     at::Tensor grad_h,
                     at::Tensor grad_tau_m,
                     at::Tensor u,
                     at::Tensor x,
                     at::Tensor h,
                     at::Tensor tau_m,
                     at::Tensor u_rest);

void fp_response_qif(at::Tensor u,
                     at::Tensor x,
                     at::Tensor h,
                     at::Tensor tau_m,
                     at::Tensor u_c,
                     at::Tensor a_0,
                     at::Tensor u_rest);

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
                     at::Tensor u_rest);

void fp_response_expif(at::Tensor u,
                       at::Tensor x,
                       at::Tensor h,
                       at::Tensor tau_m,
                       at::Tensor u_t,
                       at::Tensor delta_t,
                       at::Tensor u_rest);

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
                       at::Tensor u_rest);

void fp_spiking_heaviside(at::Tensor o, at::Tensor u, at::Tensor u_threshold);

void fp_spiking_floor(at::Tensor o,
                      at::Tensor u,
                      at::Tensor u_threshold,
                      at::Tensor u_rest);

void fp_spiking_ceil(at::Tensor o,
                     at::Tensor u,
                     at::Tensor u_threshold,
                     at::Tensor u_rest);

void fp_spiking_round(at::Tensor o,
                      at::Tensor u,
                      at::Tensor u_threshold,
                      at::Tensor u_rest);

void bp_spiking_rectangular(at::Tensor grad_o,
                            at::Tensor grad_u,
                            at::Tensor o,
                            at::Tensor u,
                            at::Tensor u_threshold,
                            float a);

void bp_spiking_polynomial(at::Tensor grad_o,
                           at::Tensor grad_u,
                           at::Tensor o,
                           at::Tensor u,
                           at::Tensor u_threshold,
                           float a);

void bp_spiking_sigmoid(at::Tensor grad_o,
                        at::Tensor grad_u,
                        at::Tensor o,
                        at::Tensor u,
                        at::Tensor u_threshold,
                        float a);

void bp_spiking_gaussian(at::Tensor grad_o,
                         at::Tensor grad_u,
                         at::Tensor o,
                         at::Tensor u,
                         at::Tensor u_threshold,
                         float a);

void bp_spiking_multi(at::Tensor grad_o,
                      at::Tensor grad_u,
                      at::Tensor o,
                      at::Tensor u,
                      at::Tensor u_threshold,
                      at::Tensor u_rest);

void fp_reset_hard(at::Tensor h, at::Tensor u, at::Tensor o, at::Tensor u_rest);

void bp_reset_hard(at::Tensor grad_h,
                   at::Tensor grad_u,
                   at::Tensor grad_o,
                   at::Tensor h,
                   at::Tensor u,
                   at::Tensor o,
                   at::Tensor u_rest);

void fp_reset_soft(at::Tensor h,
                   at::Tensor u,
                   at::Tensor o,
                   at::Tensor u_threshold,
                   at::Tensor u_rest);

void bp_reset_soft(at::Tensor grad_h,
                   at::Tensor grad_u,
                   at::Tensor grad_o,
                   at::Tensor h,
                   at::Tensor u,
                   at::Tensor o,
                   at::Tensor u_threshold,
                   at::Tensor u_rest);

void fp_lif(at::Tensor o,
            at::Tensor u,
            at::Tensor h,
            at::Tensor x,
            int time_steps,
            at::Tensor u_init,
            at::Tensor tau_m,
            at::Tensor u_rest,
            at::Tensor u_threshold,
            int firing_mode,
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
            at::Tensor u_rest,
            at::Tensor u_threshold,
            int firing_mode,
            float a,
            int reset_mode);

#endif