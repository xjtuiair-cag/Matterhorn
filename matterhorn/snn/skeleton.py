import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(self) -> None:
        """
        脉冲神经网络模块的骨架。
        """
        nn.Module.__init__(self)


    def reset(self) -> None:
        """
        重置模型。
        """
        pass


    def detach(self) -> None:
        """
        将模型中的某些变量从其计算图中分离。
        """
        pass

    
    def start_step(self) -> None:
        """
        开始STDP训练。
        """
        pass
    

    def stop_step(self) -> None:
        """
        停止STDP训练。
        """
        pass
    

    def step_once(self) -> None:
        """
        部署结点的STDP训练。
        """
        pass