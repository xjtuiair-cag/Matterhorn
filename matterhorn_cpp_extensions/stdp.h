#ifndef _MATTERHORN_STDP_H
#define _MATTERHORN_STDP_H

#include <ATen/ATen.h>
#include <vector>

void stdp(at::Tensor weight_mat,
         int input_shape,
         int output_shape,
         int time_steps,
         at::Tensor input_spike_train,
         at::Tensor output_spike_train,
         float a_pos,
         float tau_pos,
         float a_neg,
         float tau_neg,
         int batch_size);

#endif