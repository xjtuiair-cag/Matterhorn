import torch
import torch.nn as nn


from typing import Optional


"""
脉冲神经网络的容器，用来容纳时间和空间维度的脉冲神经网络集合。
建议先在空间维度上构建完整的脉冲神经网络结构，再在多个时间步之内进行模拟。
"""


class Spatial(nn.Sequential):
    def __init__(self, *args) -> None:
        """
        SNN的空间容器
        用法同nn.Sequential，加入一些特殊的作用于SNN的函数
        @params:
            *args: [nn.Module] 按空间顺序传入的各个模块
        """
        super().__init__(*args)
    

    def n_reset(self) -> None:
        """
        一次重置该序列中所有的神经元
        """
        for module in self:
            if hasattr(module, "n_reset"):
                module.n_reset()


    def start_step(self) -> None:
        """
        开始STDP训练
        """
        for module in self:
            if hasattr(module, "start_step"):
                module.start_step()


    def stop_step(self) -> None:
        """
        停止STDP训练
        """
        for module in self:
            if hasattr(module, "stop_step"):
                module.stop_step()
    

    def l_step(self) -> None:
        """
        一次部署所有结点的STDP学习
        """
        for module in self:
            if hasattr(module, "l_step"):
                module.l_step()


class Temporal(nn.Module):
    def __init__(self, model: nn.Module, reset_after_process = True) -> None:
        """
        SNN的时间容器
        在多个时间步之内执行脉冲神经网络
        @params:
            model: nn.Module 所用来执行的单步模型
            reset_after_process: bool 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__()
        self.model = model
        self.reset_after_process = reset_after_process
        self.step_after_process = False


    def n_reset(self) -> None:
        """
        重置模型
        """
        if hasattr(self.model, "n_reset"):
            self.model.n_reset()

    
    def start_step(self) -> None:
        """
        开始STDP训练
        """
        self.step_after_process = True
        if hasattr(self.model, "start_step"):
            self.model.start_step()
    

    def stop_step(self) -> None:
        """
        停止STDP训练
        """
        self.step_after_process = False
        if hasattr(self.model, "stop_step"):
            self.model.stop_step()
    

    def l_step(self) -> None:
        """
        部署结点的STDP学习
        """
        if hasattr(self.model, "l_step"):
            self.model.l_step()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，默认接受的张量形状为[T,B,...]（需要将时间维度通过permute等函数转到最外）
        @params:
            x: torch.Tensor 输入张量
        @return:
            y: torch.Tensor 输出张量
        """
        time_steps = x.shape[0]
        result = []
        for t in range(time_steps):
            result.append(self.model(x[t]))
        y = torch.stack(result)
        if self.step_after_process:
            self.l_step()
        if self.reset_after_process:
            self.n_reset()
        return y


class Container(nn.Module):
    def __init__(self, encoder: Optional[nn.Module] = None, snn_model: Optional[nn.Module] = None, decoder: Optional[nn.Module] = None) -> None:
        """
        SNN容器，包括编码器、神经网络主体和解码器，将SNN包装起来，以和ANN结合
        """
        super().__init__()
        self.encoder = encoder
        self.snn_model = snn_model
        self.decoder = decoder


    def n_reset(self) -> None:
        """
        重置模型
        """
        if hasattr(self.encoder, "n_reset"):
            self.encoder.n_reset()
        if hasattr(self.snn_model, "n_reset"):
            self.snn_model.n_reset()
        if hasattr(self.decoder, "n_reset"):
            self.decoder.n_reset()


    def start_step(self) -> None:
        """
        开始STDP训练
        """
        if hasattr(self.snn_model, "start_step"):
            self.snn_model.start_step()


    def stop_step(self) -> None:
        """
        停止STDP训练
        """
        if hasattr(self.snn_model, "stop_step"):
            self.snn_model.stop_step()
    

    def l_step(self) -> None:
        """
        部署结点的STDP学习
        """
        if hasattr(self.snn_model, "l_step"):
            self.snn_model.l_step()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，根据所传入的编解码器判断SNN中的张量形态
        @params:
            x: torch.Tensor 输入张量
        @return:
            y: torch.Tensor 输出张量
        """
        if self.encoder is not None:
            x = self.encoder(x)
        if self.snn_model is not None:
            x = self.snn_model(x)
        if self.decoder is not None:
            x = self.decoder(x)
        return x