#ifndef _MATTERHORN_SOMA_CUDA_H
#define _MATTERHORN_SOMA_CUDA_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>

void cu_fp_lif_heaviside_hard(at::Tensor o,
                              at::Tensor u,
                              at::Tensor h,
                              at::Tensor x,
                              int time_steps,
                              int shape,
                              at::Tensor u_init,
                              at::Tensor tau_m,
                              float u_rest,
                              float u_threshold);

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
                                float u_threshold);

void fp_lif_heaviside_hard_cuda(float* o,
                                float* u,
                                float* h,
                                const float* x,
                                int time_steps,
                                int shape,
                                const float* u_init,
                                const float* tau_m,
                                float u_rest,
                                float u_threshold);

void bp_lif_rectangular_hard_cuda(float* grad_o,
                                  float* grad_u,
                                  float* grad_h,
                                  float* grad_x,
                                  float* grad_u_init,
                                  float* grad_tau_m,
                                  int time_steps,
                                  int shape,
                                  const float* o,
                                  const float* u,
                                  const float* h,
                                  const float* x,
                                  const float* u_init,
                                  const float* tau_m,
                                  float u_rest,
                                  float u_threshold);

#endif