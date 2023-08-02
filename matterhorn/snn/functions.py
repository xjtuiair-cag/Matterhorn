from typing import Any


import torch
import torch.nn as nn


@torch.jit.script
def forward_heaviside(x: torch.Tensor) -> torch.Tensor:
    """
    阶跃函数。当输入大于等于0时，其输出为1；当输入小于0时，其输出为0。
    @params:
        x: torch.Tensor 输入x
    @return:
        y: torch.Tensor 输出u(x)
    """
    return x.ge(0.).to(x)


@torch.jit.script
def backward_rectangular(grad_output: torch.Tensor, x: torch.Tensor, a: float = 2.) -> torch.Tensor:
    """
    阶跃函数的导数，矩形窗，
    详见文章[Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)。
    @params:
        grad_output: torch.Tensor 输出梯度
        x: torch.Tensor 输入
        a: float 参数a，详见文章
    @return:
        grad_input: torch.Tensor 输入梯度
    """
    h = (1. / a) * torch.logical_and(x.gt(-a / 2.), x.lt(a / 2.)).to(x)
    return h * grad_output


class heaviside_rectangular(torch.autograd.Function):
    def surrogate_str(self):
        """
        反向传播的替代梯度（字符串）
        @return:
            surrogate_str: str 反向传播的替代梯度（字符串）
        """
        return "rectangular"


    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, a: float = 2.) -> torch.Tensor:
        """
        使用Heaviside阶跃函数作为前向传播函数。
        @params:
            ctx: 上下文
            x: torch.Tensor 输入
            a: float 参数a
        @return:
            y: torch.Tensor 输出
        """
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.a = a
        return forward_heaviside(x)
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        使用矩形函数作为反向传播函数。
        @params:
            ctx: 上下文
            grad_output: torch.Tensor 输出梯度
        @return:
            grad_input: torch.Tensor 输入梯度
        """
        x = ctx.saved_tensors[0]
        return backward_rectangular(grad_output, x, ctx.a), None


@torch.jit.script
def backward_polynomial(grad_output: torch.Tensor, x: torch.Tensor, a: float = 1.) -> torch.Tensor:
    """
    阶跃函数的导数，一次函数窗，
    详见文章[Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)。
    @params:
        grad_output: torch.Tensor 输出梯度
        x: torch.Tensor 输入
        a: float 参数a，详见文章
    @return:
        grad_input: torch.Tensor 输入梯度
    """
    h = ((a ** 0.5) / 2. - a / 4. * x) * torch.sign(2. / (a ** 0.5) - x)
    return h * grad_output


class heaviside_polynomial(torch.autograd.Function):
    def surrogate_str(self):
        """
        反向传播的替代梯度（字符串）
        @return:
            surrogate_str: str 反向传播的替代梯度（字符串）
        """
        return "polynomial"


    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, a: float = 1.) -> torch.Tensor:
        """
        使用Heaviside阶跃函数作为前向传播函数。
        @params:
            ctx: 上下文
            x: torch.Tensor 输入
            a: float 参数a
        @return:
            y: torch.Tensor 输出
        """
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.a = a
        return forward_heaviside(x)
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        使用多项式函数作为反向传播函数。
        @params:
            ctx: 上下文
            grad_output: torch.Tensor 输出梯度
        @return:
            grad_input: torch.Tensor 输入梯度
        """
        x = ctx.saved_tensors[0]
        return backward_polynomial(grad_output, x, ctx.a), None


@torch.jit.script
def backward_sigmoid(grad_output: torch.Tensor, x: torch.Tensor, a: float = 1.) -> torch.Tensor:
    """
    阶跃函数的导数，sigmoid函数窗，
    详见文章[Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)。
    @params:
        grad_output: torch.Tensor 输出梯度
        x: torch.Tensor 输入
        a: float 参数a，详见文章
    @return:
        grad_input: torch.Tensor 输入梯度
    """
    ex = torch.exp(-x / a)
    h = (1. / a) * (ex / ((1. + ex) ** 2.))
    return h * grad_output


class heaviside_sigmoid(torch.autograd.Function):
    def surrogate_str(self):
        """
        反向传播的替代梯度（字符串）
        @return:
            surrogate_str: str 反向传播的替代梯度（字符串）
        """
        return "sigmoid"


    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, a: float = 1.) -> torch.Tensor:
        """
        使用Heaviside阶跃函数作为前向传播函数。
        @params:
            ctx: 上下文
            x: torch.Tensor 输入
            a: float 参数a
        @return:
            y: torch.Tensor 输出
        """
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.a = a
        return forward_heaviside(x)
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        使用sigmoid函数作为反向传播函数。
        @params:
            ctx: 上下文
            grad_output: torch.Tensor 输出梯度
        @return:
            grad_input: torch.Tensor 输入梯度
        """
        x = ctx.saved_tensors[0]
        return backward_sigmoid(grad_output, x, ctx.a), None


@torch.jit.script
def backward_gaussian(grad_output: torch.Tensor, x: torch.Tensor, a: float = 1.) -> torch.Tensor:
    """
    阶跃函数的导数，高斯函数窗，
    详见文章[Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)。
    @params:
        grad_output: torch.Tensor 输出梯度
        x: torch.Tensor 输入
        a: float 参数a，详见文章
    @return:
        grad_input: torch.Tensor 输入梯度
    """
    h = (1. / ((2. * torch.pi * a) ** 0.5)) * torch.exp(-(x ** 2.) / 2 * a)
    return h * grad_output


class heaviside_gaussian(torch.autograd.Function):
    def surrogate_str(self):
        """
        反向传播的替代梯度（字符串）
        @return:
            surrogate_str: str 反向传播的替代梯度（字符串）
        """
        return "gaussian"


    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, a: float = 1.) -> torch.Tensor:
        """
        使用Heaviside阶跃函数作为前向传播函数。
        @params:
            ctx: 上下文
            x: torch.Tensor 输入
            a: float 参数a
        @return:
            y: torch.Tensor 输出
        """
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.a = a
        return forward_heaviside(x)
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        使用高斯函数作为反向传播函数。
        @params:
            ctx: 上下文
            grad_output: torch.Tensor 输出梯度
        @return:
            grad_input: torch.Tensor 输入梯度
        """
        x = ctx.saved_tensors[0]
        return backward_gaussian(grad_output, x, ctx.a), None