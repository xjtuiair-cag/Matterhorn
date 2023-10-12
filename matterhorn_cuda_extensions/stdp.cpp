#include "stdp.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>
#include <vector>

/*
STDP主函数。
@params:
    weight_mat: at::Tensor 待更新的权重矩阵，形状为[output_shape,input_shape]
    input_shape: int 输入向量长度
    output_shape: int 输出向量长度
    time_steps: int 时间步长
    input_spike_train: at::Tensor 输入脉冲序列，形状为[time_steps,input_shape]
    output_spike_train: at::Tensor 输出脉冲序列，形状为[time_steps,output_shape]
    a_pos: float STDP参数$A^{+}$
    tau_pos: float STDP参数$τ^{+}$
    a_neg: float STDP参数$A^{-}$
    tau_neg: float STDP参数$τ_{-}$
*/
void stdp(at::Tensor weight_mat,
          int input_shape,
          int output_shape,
          int time_steps,
          at::Tensor input_spike_train,
          at::Tensor output_spike_train,
          float a_pos,
          float tau_pos,
          float a_neg,
          float tau_neg) {
    float* weight_mat_head = weight_mat.data_ptr<float>();
    const float* input_spike_train_head = input_spike_train.data_ptr<float>();
    const float* output_spike_train_head = output_spike_train.data_ptr<float>();

    stdp_cuda(weight_mat_head, input_shape, output_shape, time_steps,
              input_spike_train_head, output_spike_train_head, a_pos, tau_pos,
              a_neg, tau_neg);
}