#ifndef _MATTERHORN_STDP_CUDA_H
#define _MATTERHORN_STDP_CUDA_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>

void cu_stdp(at::Tensor weight_mat,
             int input_shape,
             int output_shape,
             int time_steps,
             at::Tensor input_spike_train,
             at::Tensor output_spike_train,
             float a_pos,
             float tau_pos,
             float a_neg,
             float tau_neg);

void stdp_cuda(float* weight_mat,
               int input_shape,
               int output_shape,
               int time_steps,
               const float* input_spike_train,
               const float* output_spike_train,
               float a_pos,
               float tau_pos,
               float a_neg,
               float tau_neg);

#endif