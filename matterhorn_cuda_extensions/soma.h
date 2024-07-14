#ifndef _MATTERHORN_SOMA_CUDA_H
#define _MATTERHORN_SOMA_CUDA_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>

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
               int firing_mode,
               int reset_mode);

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
               int firing_mode,
               float a,
               int reset_mode);

void fp_lif_cuda(float* o,
                 float* u,
                 float* h,
                 float* x,
                 int time_steps,
                 int shape,
                 float* u_init,
                 float* tau_m,
                 float* u_rest,
                 float* u_threshold,
                 int firing_mode,
                 int reset_mode);

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
                 float* u_rest,
                 float* u_threshold,
                 int firing_mode,
                 float a,
                 int reset_mode);

#endif