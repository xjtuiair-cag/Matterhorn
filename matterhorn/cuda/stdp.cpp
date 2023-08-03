#include "stdp.h"
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>
#include <vector>

extern THCState* state;

int stdp(at::Tensor weight_mat,
         int input_shape,
         int output_shape,
         int time_steps,
         at::Tensor input_spike_train,
         at::Tensor output_spike_train,
         float a_pos,
         float tau_pos,
         float a_neg,
         float tau_neg) {
    const float* weight_mat_head = weight_mat.data<float>();
    const float* input_spike_train_head = input_spike_train.data<float>();
    const float* output_spike_train_head = output_spike_train.data<float>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    stdp_cuda(weight_mat_head, input_shape, output_shape, time_steps,
              input_spike_train_head, output_spike_train_head, a_pos, tau_pos,
              a_neg, tau_neg, stream);
    return 1;
}