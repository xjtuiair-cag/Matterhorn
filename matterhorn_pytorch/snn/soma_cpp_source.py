import firing as _firing
from typing import Tuple as _Tuple


__includes = """
#include <cmath>

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
void fp_response_lif(at::Tensor u, at::Tensor x, at::Tensor h, at::Tensor tau_m, at::Tensor u_rest) {
    at::Tensor du = (1.0 / tau_m) * (-(h - u_rest) + x);
    u += h + du;
}

"""


__bp_response_lif = """
void bp_response_lif(at::Tensor grad_u, at::Tensor grad_x, at::Tensor grad_h, at::Tensor grad_tau_m, at::Tensor u, at::Tensor x, at::Tensor h, at::Tensor tau_m, at::Tensor u_rest) {
    grad_x += grad_u * (1.0 / tau_m);
    grad_h += grad_u * (1.0 - (1.0 / tau_m));
    grad_tau_m += grad_u * (-1.0 / at::pow(tau_m, 2.0)) * (-(h - u_rest) + x);
}

"""


__fp_spiking_heaviside = """
void fp_spiking_heaviside(at::Tensor o, at::Tensor u, at::Tensor u_threshold) {
    o += at::ge(u, u_threshold);
}

"""


__bp_spiking_rectangular = """
void bp_spiking_rectangular(at::Tensor grad_o, at::Tensor grad_u, at::Tensor o, at::Tensor u, at::Tensor u_threshold, float a = 2.0) {
    at::Tensor ax = u - u_threshold;
    grad_u += grad_o * (1.0 / a) * at::lt(at::abs(ax), a / 2.0);
}

"""


__bp_spiking_polynomial = """
void bp_spiking_polynomial(at::Tensor grad_o, at::Tensor grad_u, at::Tensor o, at::Tensor u, at::Tensor u_threshold, float a = 1.0) {
    at::Tensor ax = at::abs(u - u_threshold);
    grad_u += grad_o * (sqrtf(a) / 2.0 - a / 4.0 * ax) * at::sign(2.0 / sqrtf(a) - ax) * at::lt(ax, 2.0 / sqrtf(a));
}

"""


__bp_spiking_sigmoid = """
void bp_spiking_sigmoid(at::Tensor grad_o, at::Tensor grad_u, at::Tensor o, at::Tensor u, at::Tensor u_threshold, float a = 1.0) {
    at::Tensor ax = u - u_threshold;
    at::Tensor ex = at::exp(-ax / a);
    grad_u += grad_o * (1.0 / a) * ex / at::pow(1.0 + ex, 2.0);
}

"""


__bp_spiking_gaussian = """
void bp_spiking_gaussian(at::Tensor grad_o, at::Tensor grad_u, at::Tensor o, at::Tensor u, at::Tensor u_threshold, float a = 1.0) {
    at::Tensor ax = u - u_threshold;
    grad_u += grad_o / sqrtf(2.0 * M_PI * a) * at::exp(-at::pow(ax, 2.0) / (2.0 * a));
}

"""


__fp_spiking_floor = """
void fp_spiking_floor(at::Tensor o, at::Tensor u, at::Tensor u_threshold, at::Tensor u_rest) {
    o += at::floor((u - u_rest) / (u_threshold - u_rest));
}

"""


__fp_spiking_ceil = """
void fp_spiking_ceil(at::Tensor o, at::Tensor u, at::Tensor u_threshold, at::Tensor u_rest) {
    o += at::ceil((u - u_rest) / (u_threshold - u_rest));
}

"""


__fp_spiking_round = """
void fp_spiking_round(at::Tensor o, at::Tensor u, at::Tensor u_threshold, at::Tensor u_rest) {
    o += at::round((u - u_rest) / (u_threshold - u_rest));
}

"""


__bp_spiking_multi = """
void bp_spiking_multi(at::Tensor grad_o, at::Tensor grad_u, at::Tensor o, at::Tensor u, at::Tensor u_threshold, at::Tensor u_rest) {
    grad_u += (u - u_rest) / (u_threshold - u_rest);
}

"""


__fp_reset_hard = """
void fp_reset_hard(at::Tensor h, at::Tensor u, at::Tensor o, at::Tensor u_rest) {
    h += u * (1.0 - o) + u_rest * o;
}

"""


__bp_reset_hard = """
void bp_reset_hard(at::Tensor grad_h, at::Tensor grad_u, at::Tensor grad_o, at::Tensor h, at::Tensor u, at::Tensor o, at::Tensor u_rest) {
    grad_u += grad_h * (1.0 - o);
    grad_o += grad_h * (u_rest - u);
}

"""


__fp_reset_soft = """
void fp_reset_soft(at::Tensor h, at::Tensor u, at::Tensor o, at::Tensor u_threshold, at::Tensor u_rest) {
    h += u - (u_threshold - u_rest) * o;
}

"""


__bp_reset_soft = """
void bp_reset_soft(at::Tensor grad_h, at::Tensor grad_u, at::Tensor grad_o, at::Tensor h, at::Tensor u, at::Tensor o, at::Tensor u_threshold, at::Tensor u_rest) {
    grad_u += grad_h;
    grad_o += grad_h * -(u_threshold - u_rest);
}

"""


def __multi_spiking(spiking_function: _firing.Firing) -> bool:
    if isinstance(spiking_function, (_firing.Floor, _firing.Ceil, _firing.Round)):
        return True
    elif isinstance(spiking_function, (_firing.Rectangular, _firing.Polynomial, _firing.Sigmoid, _firing.Gaussian)):
        return False
    else:
        raise ValueError("Unknown spiking function: %s" % (spiking_function.__class__.__name__,))


def __fp_spiking_source(spiking_function: _firing.Firing) -> _Tuple[str, str]:
    if not __multi_spiking(spiking_function):
        return "fp_spiking_heaviside(o[t], u[t], u_threshold);", __fp_spiking_heaviside
    elif isinstance(spiking_function, (_firing.Floor,)):
        return "fp_spiking_floor(o[t], u[t], u_threshold, u_rest);", __fp_spiking_floor
    elif isinstance(spiking_function, (_firing.Ceil,)):
        return "fp_spiking_ceil(o[t], u[t], u_threshold, u_rest);", __fp_spiking_ceil
    elif isinstance(spiking_function, (_firing.Round,)):
        return "fp_spiking_round(o[t], u[t], u_threshold, u_rest);", __fp_spiking_round
    else:
        raise ValueError("Unknown spiking function: %s" % (spiking_function.__class__.__name__,))


def __fp_reset_source(hard_reset: bool, multi_spiking: bool) -> _Tuple[str, str]:
    if hard_reset:
        return "fp_reset_hard(h[t], u[t], o[t], u_rest);", __fp_reset_hard
    else:
        return "fp_reset_soft(h[t], u[t], o[t], u_threshold, u_rest);", __fp_reset_soft


def __bp_spiking_source(spiking_function: _firing.Firing) -> _Tuple[str, str]:
    if __multi_spiking(spiking_function):
        return "bp_spiking_multi(grad_o[t], grad_u[t], o[t], u[t], u_threshold, u_rest);", __bp_spiking_multi
    elif isinstance(spiking_function, (_firing.Rectangular,)):
        return "bp_spiking_rectangular(grad_o[t], grad_u[t], o[t], u[t], u_threshold, %g);" % (spiking_function.a), __bp_spiking_rectangular
    elif isinstance(spiking_function, (_firing.Polynomial,)):
        return "bp_spiking_polynomial(grad_o[t], grad_u[t], o[t], u[t], u_threshold, %g);" % (spiking_function.a), __bp_spiking_polynomial
    elif isinstance(spiking_function, (_firing.Sigmoid,)):
        return "bp_spiking_sigmoid(grad_o[t], grad_u[t], o[t], u[t], u_threshold, %g);" % (spiking_function.a), __bp_spiking_sigmoid
    elif isinstance(spiking_function, (_firing.Gaussian,)):
        return "bp_spiking_gaussian(grad_o[t], grad_u[t], o[t], u[t], u_threshold, %g);" % (spiking_function.a), __bp_spiking_gaussian
    else:
        raise ValueError("Unknown spiking function: %s" % (spiking_function.__class__.__name__,))


def __bp_reset_source(hard_reset: bool, multi_spiking: bool) -> _Tuple[str, str]:
    if hard_reset:
        return "bp_reset_hard(grad_h[t], grad_u[t], grad_o[t], h[t], u[t], o[t], u_rest);", __bp_reset_hard
    else:
        return "bp_reset_soft(grad_h[t], grad_u[t], grad_o[t], h[t], u[t], o[t], u_threshold, u_rest);", __bp_reset_soft


__fp_lif = """
void fp_lif(int time_steps, at::Tensor o, at::Tensor u, at::Tensor h, at::Tensor x, at::Tensor u_init, at::Tensor tau_m, at::Tensor u_rest, at::Tensor u_threshold) {
    for (int t = 0; t < time_steps; t++) {
        fp_response_lif(u[t], x[t], t ? h[t - 1] : u_init, tau_m, u_rest);
        %s
        %s
    }
}

"""


__bp_lif = """
void bp_lif(int time_steps, at::Tensor grad_o, at::Tensor grad_u, at::Tensor grad_h, at::Tensor grad_x, at::Tensor grad_u_init, at::Tensor grad_tau_m, at::Tensor o, at::Tensor u, at::Tensor h, at::Tensor x, at::Tensor u_init, at::Tensor tau_m, at::Tensor u_rest, at::Tensor u_threshold) {
    for (int t = time_steps - 1; t >= 0; t--) {
        %s
        %s
        bp_response_lif(grad_u[t], grad_x[t], t ? grad_h[t - 1] : grad_u_init, grad_tau_m, u[t], x[t], t ? h[t - 1] : u_init, tau_m, u_rest);
    }
}

"""


def __fp_lif_source(spiking_function: _firing.Firing, hard_reset: bool) -> _Tuple[str, str]:
    spiking_fun, spiking_source = __fp_spiking_source(spiking_function)
    reset_fun, reset_source = __fp_reset_source(hard_reset, __multi_spiking(spiking_function))
    res = __includes + __fp_response_lif + spiking_source + reset_source + (__fp_lif % (spiking_fun, reset_fun))
    return "fp_lif", res


def __bp_lif_source(spiking_function: _firing.Firing, hard_reset: bool) -> _Tuple[str, str]:
    spiking_fun, spiking_source = __bp_spiking_source(spiking_function)
    reset_fun, reset_source = __bp_reset_source(hard_reset, __multi_spiking(spiking_function))
    res = __includes + __bp_response_lif + spiking_source + reset_source + (__bp_lif % (reset_fun, spiking_fun))
    return "bp_lif", res