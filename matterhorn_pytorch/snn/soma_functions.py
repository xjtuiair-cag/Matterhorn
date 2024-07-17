import torch
from typing import Any as _Any, Callable as _Callable


class multi_step_mode_lif(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Any, x: torch.Tensor, u_init: torch.Tensor, u_threshold: torch.Tensor, u_rest: torch.Tensor, tau_m: torch.Tensor, fp_lif: _Callable, bp_lif: _Callable) -> torch.Tensor:
        ctx.time_steps = x.shape[0]
        ctx.shape = torch.prod(torch.tensor(x.shape[1:])).item()
        ctx.fp_lif = fp_lif
        ctx.bp_lif = bp_lif
        o = torch.zeros_like(x)
        u = torch.zeros_like(x)
        h = torch.zeros_like(x)
        ctx.fp_lif(ctx.time_steps, ctx.shape, o, u, h, x, u_init, u_threshold, u_rest, tau_m)
        ctx.save_for_backward(o, u, h, x, u_init, u_threshold, u_rest, tau_m)
        u_last = u[-1]
        return o, u_last
    

    @staticmethod
    def backward(ctx: _Any, grad_o: torch.Tensor, grad_u_last: torch.Tensor) -> torch.Tensor:
        grad_o = grad_o.clone()
        o, u, h, x, u_init, u_threshold, u_rest, tau_m = ctx.saved_tensors
        grad_u = torch.zeros_like(u)
        grad_h = torch.zeros_like(h)
        grad_h[-1] = grad_u_last
        grad_x = torch.zeros_like(x)
        grad_u_init = torch.zeros_like(u_init)
        grad_u_threshold = torch.zeros_like(u_threshold)
        grad_u_rest = torch.zeros_like(u_rest)
        grad_tau_m = torch.zeros_like(u_init)
        ctx.bp_lif(ctx.time_steps, ctx.shape, grad_o, grad_u, grad_h, grad_x, grad_u_init, grad_tau_m, o, u, h, x, u_init, u_threshold, u_rest, tau_m)
        grad_tau_m = torch.sum(grad_tau_m)
        return grad_x, grad_u_init, grad_u_threshold, grad_u_rest, grad_tau_m, None, None