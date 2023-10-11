#include "soma.h"
#include <torch/serialize/tensor.h>
#include <cmath>
#include <vector>

int lif_response_fp(at::Tensor u,
                    at::Tensor x,
                    at::Tensor h,
                    int shape,
                    at::Tensor tau_m,
                    float u_rest) {
                        for(int i = 0; i < shape; i++) {

                        }
                    }

int lif_spiking_fp(at::Tensor o, at::Tensor u, int shape, float u_threshold) {}

int lif_reset_fp(at::Tensor h,
                 at::Tensor u,
                 at::Tensor o,
                 int shape,
                 float u_rest) {}

int lif_response_bp(at::Tensor grad_u,
                    at::Tensor grad_x,
                    at::Tensor grad_h,
                    int shape,
                    at::Tensor tau_m,
                    float u_rest) {}

int lif_spiking_bp(at::Tensor grad_o,
                   at::Tensor grad_u,
                   int shape,
                   float u_threshold) {}

int lif_reset_bp(at::Tensor grad_h,
                 at::Tensor grad_u,
                 at::Tensor grad_o,
                 int shape,
                 float u_rest) {}


/*
LIF神经元的前向传播函数
@params:
    o: at::Tensor 脉冲输出o
    
*/
int lif_fp(at::Tensor o,
           at::Tensor u,
           at::Tensor x,
           int shape,
           int time_steps,
           at::Tensor tau_m,
           float u_rest,
           float u_threshold) {}

int lif_bp(at::Tensor grad_o,
           at::Tensor grad_u,
           at::Tensor grad_x,
           int shape,
           int time_steps,
           at::Tensor tau_m,
           float u_rest,
           float u_threshold) {}