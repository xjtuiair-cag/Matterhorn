import torch
from typing import Any as _Any, Tuple as _Tuple, Callable as _Callable


class oo(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor) -> torch.Tensor:
        return x.gt(0.0).to(x)
    

    @staticmethod
    def backward(ctx: _Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def multi_step_mode_lif_fp_ext(fp: _Callable, x: torch.Tensor, u_init: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, tau_m: torch.Tensor) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = x.shape[0]
    L = torch.prod(torch.tensor(x.shape[1:])).item()
    o = torch.zeros_like(x)
    u = torch.zeros_like(x)
    h = torch.zeros_like(x)
    fp(T, L, o, u, h, x, u_init, u_threshold, u_rest, tau_m)
    return o, u, h


def multi_step_mode_lif_bp_ext(fp: _Callable, bp: _Callable, grad_o: torch.Tensor, grad_u_last: torch.Tensor, x: torch.Tensor, u_init: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, tau_m: torch.Tensor) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = x.shape[0]
    L = torch.prod(torch.tensor(x.shape[1:])).item()
    o, u, h = multi_step_mode_lif_fp_ext(fp, x, u_init, u_threshold, u_rest, tau_m)
    grad_u = torch.zeros_like(u)
    grad_h = torch.zeros_like(h)
    grad_h[-1] = grad_u_last
    grad_x = torch.zeros_like(x)
    grad_u_init = torch.zeros_like(u_init)
    grad_tau_m = torch.zeros_like(u_init)
    bp(T, L, grad_o, grad_u, grad_h, grad_x, grad_u_init, grad_tau_m, o, u, h, x, u_init, u_threshold, u_rest, tau_m)
    grad_tau_m = torch.sum(grad_tau_m)
    return grad_x, grad_u_init, grad_tau_m


class multi_step_mode_lif(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor, u_init: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, tau_m: torch.Tensor, fp_lif: _Callable, bp_lif: _Callable) -> torch.Tensor:
        ctx.fp_lif = fp_lif
        ctx.bp_lif = bp_lif
        o, u, h = multi_step_mode_lif_fp_ext(fp_lif, x, u_init, u_threshold, u_rest, tau_m)
        ctx.save_for_backward(x, u_init, u_threshold, u_rest, tau_m)
        u_last = u[-1].clone()
        return o, u_last
    

    @staticmethod
    def backward(ctx: _Any, grad_o: torch.Tensor, grad_u_last: torch.Tensor) -> torch.Tensor:
        grad_o = grad_o.clone()
        x, u_init, u_threshold, u_rest, tau_m = ctx.saved_tensors
        grad_x, grad_u_init, grad_tau_m = multi_step_mode_lif_bp_ext(ctx.fp_lif, ctx.bp_lif, grad_o, grad_u_last, x, u_init, u_threshold, u_rest, tau_m)
        grad_u_threshold = torch.zeros_like(u_threshold)
        grad_u_rest = torch.zeros_like(u_rest)
        return grad_x, grad_u_init, grad_u_threshold, grad_u_rest, grad_tau_m, None, None


def multi_step_mode_liaf_fp_ext(fp: _Callable, x: torch.Tensor, u_init: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, tau_m: torch.Tensor) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = x.shape[0]
    L = torch.prod(torch.tensor(x.shape[1:])).item()
    o = torch.zeros_like(x)
    u = torch.zeros_like(x)
    h = torch.zeros_like(x)
    fp(T, L, o, u, h, x, u_init, u_threshold, u_rest, tau_m)
    return o, u, h


def multi_step_mode_liaf_bp_ext(fp: _Callable, bp: _Callable, grad_u: torch.Tensor, grad_u_last: torch.Tensor, x: torch.Tensor, u_init: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, tau_m: torch.Tensor) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = x.shape[0]
    L = torch.prod(torch.tensor(x.shape[1:])).item()
    o, u, h = multi_step_mode_liaf_fp_ext(fp, x, u_init, u_threshold, u_rest, tau_m)
    grad_o = torch.zeros_like(o)
    grad_h = torch.zeros_like(h)
    grad_h[-1] = grad_u_last
    grad_x = torch.zeros_like(x)
    grad_u_init = torch.zeros_like(u_init)
    grad_tau_m = torch.zeros_like(u_init)
    bp(T, L, grad_o, grad_u, grad_h, grad_x, grad_u_init, grad_tau_m, o, u, h, x, u_init, u_threshold, u_rest, tau_m)
    grad_tau_m = torch.sum(grad_tau_m)
    return grad_x, grad_u_init, grad_tau_m


class multi_step_mode_liaf(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor, u_init: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, tau_m: torch.Tensor, fp_lif: _Callable, bp_lif: _Callable) -> torch.Tensor:
        ctx.fp_lif = fp_lif
        ctx.bp_lif = bp_lif
        o, u, h = multi_step_mode_lif_fp_ext(fp_lif, x, u_init, u_threshold, u_rest, tau_m)
        ctx.save_for_backward(x, u_init, u_threshold, u_rest, tau_m)
        u_last = u[-1].clone()
        return u, u_last
    

    @staticmethod
    def backward(ctx: _Any, grad_u: torch.Tensor, grad_u_last: torch.Tensor) -> torch.Tensor:
        grad_u = grad_u.clone()
        x, u_init, u_threshold, u_rest, tau_m = ctx.saved_tensors
        grad_x, grad_u_init, grad_tau_m = multi_step_mode_lif_bp_ext(ctx.fp_lif, ctx.bp_lif, grad_u, grad_u_last, x, u_init, u_threshold, u_rest, tau_m)
        grad_u_threshold = torch.zeros_like(u_threshold)
        grad_u_rest = torch.zeros_like(u_rest)
        return grad_x, grad_u_init, grad_u_threshold, grad_u_rest, grad_tau_m, None, None