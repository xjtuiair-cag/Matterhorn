import matterhorn_pytorch.snn.firing as _firing
from matterhorn_pytorch._ext.functional import *
from typing import Tuple as _Tuple, Optional as _Optional


__includes = """
#include <cmath>
#include <stdlib.h>

#ifndef _MTH_VARIABLES
#define _MTH_VARIABLES

#define THREADS_PER_BLOCK 1024
#define div_ceil(base, comp) (base / comp + (base % comp ? 1 : 0))

#endif

"""


__functions = """
#ifndef _MTH_FUNCTIONS
#define _MTH_FUNCTIONS

__device__ float absf(float base) {
    return base >= 0.0 ? base : -base;
}

__device__ float sgnf(float base) {
    return base != 0.0 ? (base > 0.0 ? 1.0 : -1.0) : 0.0;
}

__device__ float logical_notf(float base) {
    return base ? 0.0 : 1.0;
}

__device__ float eqf(float base, float comp) {
    return base == comp ? 1.0 : 0.0;
}

__device__ float nef(float base, float comp) {
    return base != comp ? 1.0 : 0.0;
}

__device__ float ltf(float base, float comp) {
    return base < comp ? 1.0 : 0.0;
}

__device__ float lef(float base, float comp) {
    return base <= comp ? 1.0 : 0.0;
}

__device__ float gtf(float base, float comp) {
    return base > comp ? 1.0 : 0.0;
}

__device__ float gef(float base, float comp) {
    return base >= comp ? 1.0 : 0.0;
}

__device__ float logical_andf(float base, float comp) {
    return base && comp ? 1.0 : 0.0;
}

__device__ float logical_orf(float base, float comp) {
    return base || comp ? 1.0 : 0.0;
}

__device__ float logical_xorf(float base, float comp) {
    return ((base || comp) && !(base && comp)) ? 1.0 : 0.0;
}

__device__ float winnbf(float base, float min, float max) {
    return ((base > min) && (base < max)) ? 1.0 : 0.0;
}

__device__ float winlbf(float base, float min, float max) {
    return ((base >= min) && (base < max)) ? 1.0 : 0.0;
}

__device__ float winrbf(float base, float min, float max) {
    return ((base > min) && (base <= max)) ? 1.0 : 0.0;
}

__device__ float winbf(float base, float min, float max) {
    return ((base >= min) && (base <= max)) ? 1.0 : 0.0;
}

__device__ float clampf(float base, float min, float max) {
    return base > min ? (base < max ? base : max) : min;
}

#endif

"""


__fp_response_if = """
void fp_response_if(at::Tensor u, at::Tensor x, at::Tensor h) {
    at::Tensor du = x;
    u += h + du;
}

"""


__bp_response_if = """
void bp_response_if(at::Tensor grad_u, at::Tensor grad_x, at::Tensor grad_h, at::Tensor u, at::Tensor x, at::Tensor h) {
    grad_x += grad_u;
    grad_h += grad_u;
}

"""


__fp_response_lif = """
__device__ void fp_response_lif(float& u, float x, float h, float tau_m, float u_rest) {
    u = h + (1.0 / tau_m) * (-(h - u_rest) + x);
}

"""


__bp_response_lif = """
__device__ void bp_response_lif(float grad_u, float& grad_x, float& grad_h, float& grad_tau_m, float u, float x, float h, float tau_m, float u_rest) {
    grad_x += grad_u * (1.0 / tau_m);
    grad_h = grad_u * (1.0 - (1.0 / tau_m));
    grad_tau_m += grad_u * (-1.0 / powf(tau_m, 2.0)) * (-(h - u_rest) + x);
}

"""


__fp_spiking_heaviside = """
__device__ void fp_spiking_heaviside(float& o, float u, float u_threshold, float u_rest) {
    o = gef(u, u_threshold);
}

"""


__bp_spiking_rectangular = """
__device__ void bp_spiking_rectangular(float grad_o, float& grad_u, float o, float u, float u_threshold, float u_rest, float a) {
    float ax = u - u_threshold;
    grad_u += grad_o * (1.0 / a) * ltf(absf(ax), a / 2.0);
}

"""


__bp_spiking_polynomial = """
__device__ void bp_spiking_polynomial(float grad_o, float& grad_u, float o, float u, float u_threshold, float u_rest, float a) {
    float ax = absf(u - u_threshold);
    grad_u += grad_o * (sqrtf(a) / 2.0 - a / 4.0 * ax) * sgnf(2.0 / sqrtf(a) - ax) * ltf(ax, 2.0 / sqrtf(a));
}

"""


__bp_spiking_sigmoid = """
__device__ void bp_spiking_sigmoid(float grad_o, float& grad_u, float o, float u, float u_threshold, float u_rest, float a) {
    float ax = u - u_threshold;
    float ex = expf(-ax / a);
    grad_u += grad_o * (1.0 / a) * ex / powf(1.0 + ex, 2.0);
}

"""


__bp_spiking_gaussian = """
__device__ void bp_spiking_gaussian(float grad_o, float& grad_u, float o, float u, float u_threshold, float u_rest, float a) {
    float ax = u - u_threshold;
    grad_u += grad_o / sqrtf(2.0 * M_PI * a) * expf(-powf(ax, 2.0) / (2.0 * a));
}

"""


__fp_spiking_floor = """
__device__ void fp_spiking_floor(float& o, float u, float u_threshold, float u_rest) {
    o = floorf((u - u_rest) / (u_threshold - u_rest));
}

"""


__fp_spiking_ceil = """
__device__ void fp_spiking_ceil(float& o, float u, float u_threshold, float u_rest) {
    o = ceilf((u - u_rest) / (u_threshold - u_rest));
}

"""


__fp_spiking_round = """
__device__ void fp_spiking_round(float& o, float u, float u_threshold, float u_rest) {
    o = roundf((u - u_rest) / (u_threshold - u_rest));
}

"""


__bp_spiking_multi = """
__device__ void bp_spiking_multi(float grad_o, float& grad_u, float o, float u, float u_threshold, float u_rest) {
    grad_u += (u - u_rest) / (u_threshold - u_rest);
}

"""


__fp_reset_hard = """
__device__ void fp_reset_hard(float& h, float u, float o, float u_threshold, float u_rest) {
    h = u * (1.0 - o) + u_rest * o;
}

"""


__bp_reset_hard = """
__device__ void bp_reset_hard(float grad_h, float& grad_u, float& grad_o, float h, float u, float o, float u_threshold, float u_rest) {
    grad_u += grad_h * (1.0 - o);
    grad_o += grad_h * (u_rest - u);
}

"""


__fp_reset_soft = """
__device__ void fp_reset_soft(float& h, float u, float o, float u_threshold, float u_rest) {
    h = u - (u_threshold - u_rest) * o;
}

"""


__bp_reset_soft = """
__device__ void bp_reset_soft(float grad_h, float& grad_u, float& grad_o, float h, float u, float o, float u_threshold, float u_rest) {
    grad_u += grad_h;
    grad_o += grad_h * -(u_threshold - u_rest);
}

"""


def __fp_spiking_source(spiking_function: _firing.Firing) -> _Tuple[str, str]:
    if not multi_spiking(spiking_function):
        return "fp_spiking_heaviside(o[cur_idx], u[cur_idx], u_threshold[0], u_rest[0]);", __fp_spiking_heaviside
    elif isinstance(spiking_function, (_firing.Floor,)):
        return "fp_spiking_floor(o[cur_idx], u[cur_idx], u_threshold[0], u_rest[0]);", __fp_spiking_floor
    elif isinstance(spiking_function, (_firing.Ceil,)):
        return "fp_spiking_ceil(o[cur_idx], u[cur_idx], u_threshold[0], u_rest[0]);", __fp_spiking_ceil
    elif isinstance(spiking_function, (_firing.Round,)):
        return "fp_spiking_round(o[cur_idx], u[cur_idx], u_threshold[0], u_rest[0]);", __fp_spiking_round
    else:
        raise ValueError("Unknown spiking function: %s" % (spiking_function.__class__.__name__,))


def __fp_reset_source(hard_reset: bool, multi_spiking: bool) -> _Tuple[str, str]:
    if hard_reset:
        return "fp_reset_hard(h[cur_idx], u[cur_idx], o[cur_idx], u_threshold[0], u_rest[0]);", __fp_reset_hard
    else:
        return "fp_reset_soft(h[cur_idx], u[cur_idx], o[cur_idx], u_threshold[0], u_rest[0]);", __fp_reset_soft


def __bp_spiking_source(spiking_function: _firing.Firing) -> _Tuple[str, str]:
    if multi_spiking(spiking_function):
        return "bp_spiking_multi(grad_o[cur_idx], grad_u[cur_idx], o[cur_idx], u[cur_idx], u_threshold[0], u_rest[0]);", __bp_spiking_multi
    elif isinstance(spiking_function, (_firing.Rectangular,)):
        return "bp_spiking_rectangular(grad_o[cur_idx], grad_u[cur_idx], o[cur_idx], u[cur_idx], u_threshold[0], u_rest[0], %g);" % (spiking_function.a), __bp_spiking_rectangular
    elif isinstance(spiking_function, (_firing.Polynomial,)):
        return "bp_spiking_polynomial(grad_o[cur_idx], grad_u[cur_idx], o[cur_idx], u[cur_idx], u_threshold[0], u_rest[0], %g);" % (spiking_function.a), __bp_spiking_polynomial
    elif isinstance(spiking_function, (_firing.Sigmoid,)):
        return "bp_spiking_sigmoid(grad_o[cur_idx], grad_u[cur_idx], o[cur_idx], u[cur_idx], u_threshold[0], u_rest[0], %g);" % (spiking_function.a), __bp_spiking_sigmoid
    elif isinstance(spiking_function, (_firing.Gaussian,)):
        return "bp_spiking_gaussian(grad_o[cur_idx], grad_u[cur_idx], o[cur_idx], u[cur_idx], u_threshold[0], u_rest[0], %g);" % (spiking_function.a), __bp_spiking_gaussian
    else:
        raise ValueError("Unknown spiking function: %s" % (spiking_function.__class__.__name__,))


def __bp_reset_source(hard_reset: bool, multi_spiking: bool) -> _Tuple[str, str]:
    if hard_reset:
        return "bp_reset_hard(cur_grad_h, grad_u[cur_idx], grad_o[cur_idx], h[cur_idx], u[cur_idx], o[cur_idx], u_threshold[0], u_rest[0]);", __bp_reset_hard
    else:
        return "bp_reset_soft(cur_grad_h, grad_u[cur_idx], grad_o[cur_idx], h[cur_idx], u[cur_idx], o[cur_idx], u_threshold[0], u_rest[0]);", __bp_reset_soft


__fp_soma = """
__global__ void fp_%s_cuda_kernel(int time_steps, int shape, float* o, float* u, float* h, float* x, float* u_init, float* u_threshold, float* u_rest%s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= shape) {
        return;
    }
    for (int t = 0; t < time_steps; t++) {
        const int cur_idx = t * shape + idx;
        float last_h = t ? h[cur_idx - shape] : u_init[idx];
        %s
        %s
        %s
    }
}

void fp_%s_cuda(int time_steps, int shape, at::Tensor o, at::Tensor u, at::Tensor h, at::Tensor x, at::Tensor u_init, at::Tensor u_threshold, at::Tensor u_rest%s) {
    cudaError_t err;
    dim3 blocks(div_ceil(shape, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    fp_%s_cuda_kernel<<<blocks, threads, 0>>>(time_steps, shape, o.data_ptr<float>(), u.data_ptr<float>(), h.data_ptr<float>(), x.data_ptr<float>(), u_init.data_ptr<float>(), u_threshold.data_ptr<float>(), u_rest.data_ptr<float>()%s);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %%s\\n", cudaGetErrorString(err));
        exit(-1);
    }
}

"""


__fp_soma_declaration = """
void fp_%s_cuda(int time_steps, int shape, at::Tensor o, at::Tensor u, at::Tensor h, at::Tensor x, at::Tensor u_init, at::Tensor u_threshold, at::Tensor u_rest%s);

"""


def fp_soma_source(soma_name: str, f_response: str, f_firing: str, f_reset: str, param_table: _Optional[_Tuple[str]] = None) -> _Tuple[str, str]:
    shell_params = ""
    kernel_params = ""
    call_params = ""
    if param_table is not None:
        for param in param_table:
            shell_params += ", at::Tensor " + param
            kernel_params += ", float* " + param
            call_params += ", " + param + ".data_ptr<float>()"
    res = __fp_soma % (soma_name, kernel_params, f_response, f_firing, f_reset, soma_name, shell_params, soma_name, call_params)
    dec = __fp_soma_declaration % (soma_name, shell_params)
    return "fp_%s_cuda" % (soma_name,), dec, res


__bp_soma = """
__global__ void bp_%s_cuda_kernel(int time_steps, int shape, float* grad_o, float* grad_u, float* grad_h, float* grad_x, float* grad_u_init%s, float* o, float* u, float* h, float* x, float* u_init, float* u_threshold, float* u_rest%s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= shape) {
        return;
    }
    float cur_grad_h = 0.0f;
    for (int t = time_steps - 1; t >= 0; t--) {
        const int cur_idx = t * shape + idx;
        float last_h = t ? h[cur_idx - shape] : u_init[idx];
        %s
        %s
        %s
        if (t) {
            grad_h[cur_idx - shape] = cur_grad_h;
        } else {
            grad_u_init[idx] = cur_grad_h;
        }
    }
}

void bp_%s_cuda(int time_steps, int shape, at::Tensor grad_o, at::Tensor grad_u, at::Tensor grad_h, at::Tensor grad_x, at::Tensor grad_u_init%s, at::Tensor o, at::Tensor u, at::Tensor h, at::Tensor x, at::Tensor u_init, at::Tensor u_threshold, at::Tensor u_rest%s) {
    cudaError_t err;
    dim3 blocks(div_ceil(shape, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    bp_%s_cuda_kernel<<<blocks, threads, 0>>>(time_steps, shape, grad_o.data_ptr<float>(), grad_u.data_ptr<float>(), grad_h.data_ptr<float>(), grad_x.data_ptr<float>(), grad_u_init.data_ptr<float>()%s, o.data_ptr<float>(), u.data_ptr<float>(), h.data_ptr<float>(), x.data_ptr<float>(), u_init.data_ptr<float>(), u_threshold.data_ptr<float>(), u_rest.data_ptr<float>()%s);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %%s\\n", cudaGetErrorString(err));
        exit(-1);
    }
}

"""


__bp_soma_declaration = """
void bp_%s_cuda(int time_steps, int shape, at::Tensor grad_o, at::Tensor grad_u, at::Tensor grad_h, at::Tensor grad_x, at::Tensor grad_u_init%s, at::Tensor o, at::Tensor u, at::Tensor h, at::Tensor x, at::Tensor u_init, at::Tensor u_threshold, at::Tensor u_rest%s);

"""


def bp_soma_source(soma_name: str, grad_reset: str, grad_firing: str, grad_response: str, param_table: _Optional[_Tuple[str]] = None) -> _Tuple[str, str]:
    shell_grad_params = ""
    shell_params = ""
    kernel_grad_params = ""
    kernel_params = ""
    call_grad_params = ""
    call_params = ""
    if param_table is not None:
        for param in param_table:
            shell_grad_params += ", at::Tensor grad_" + param
            shell_params += ", at::Tensor " + param
            kernel_grad_params += ", float* grad_" + param
            kernel_params += ", float* " + param
            call_grad_params += ", grad_" + param + ".data_ptr<float>()"
            call_params += ", " + param + ".data_ptr<float>()"
    res = __bp_soma % (soma_name, kernel_grad_params, kernel_params, grad_reset, grad_firing, grad_response, soma_name, shell_grad_params, shell_params, soma_name, call_grad_params, call_params)
    dec = __bp_soma_declaration % (soma_name, shell_grad_params, shell_params)
    return "bp_%s_cuda" % (soma_name,), dec, res


def fp_lif_source(spiking_function: _firing.Firing, hard_reset: bool) -> _Tuple[str, str]:
    response_fun = "fp_response_lif(u[cur_idx], x[cur_idx], last_h, tau_m[0], u_rest[0]);"
    spiking_fun, spiking_source = __fp_spiking_source(spiking_function)
    reset_fun, reset_source = __fp_reset_source(hard_reset, multi_spiking(spiking_function))
    name, dec, __fp_lif = fp_soma_source("lif", response_fun, spiking_fun, reset_fun, ("tau_m",))
    res = __includes + __functions + __fp_response_lif + spiking_source + reset_source + __fp_lif
    return name, dec, res


def bp_lif_source(spiking_function: _firing.Firing, hard_reset: bool) -> _Tuple[str, str]:
    response_fun = "bp_response_lif(grad_u[cur_idx], grad_x[cur_idx], cur_grad_h, grad_tau_m[idx], u[cur_idx], x[cur_idx], last_h, tau_m[0], u_rest[0]);"
    spiking_fun, spiking_source = __bp_spiking_source(spiking_function)
    reset_fun, reset_source = __bp_reset_source(hard_reset, multi_spiking(spiking_function))
    name, dec, __bp_lif = bp_soma_source("lif", reset_fun, spiking_fun, response_fun, ("tau_m",))
    res = __includes + __functions + __bp_response_lif + spiking_source + reset_source + __bp_lif
    return name, dec, res


if __name__ == "__main__":
    from rich import print
    title, body = bp_lif_source(_firing.Gaussian(), True)
    print(title)
    print(body)