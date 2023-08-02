import torch
import torch.nn as nn


"""
脉冲神经网络神经元的突触，一层的前半段。输入为脉冲，输出为模拟电位值。
由突触将来自上一层神经元的脉冲信号$O_{j}^{l-1}(t)$整合成为突触后电位$X_{i}^{l}(t)$后，在胞体中进行突触后电位的累积和发放。
"""


from torch.nn import Linear as Linear


from torch.nn import Conv1d as Conv1d


from torch.nn import Conv2d as Conv2d


from torch.nn import Conv3d as Conv3d


from torch.nn import ConvTranspose1d as ConvTranspose1d


from torch.nn import ConvTranspose2d as ConvTranspose2d


from torch.nn import ConvTranspose3d as ConvTranspose3d