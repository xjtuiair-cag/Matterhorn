# -*- coding: UTF-8 -*-
"""
此文件夹放置SNN的一些基本模块。
"""


from .container import Spatial, Temporal, Sequential
from .decoder import SumSpike as SumSpikeDecoder, AverageSpike as AvgSpikeDecoder, MinTime as MinTimeDecoder, AverageTime as AvgTimeDecoder
from .encoder import Direct as DirectEncoder, Poisson as PoissonEncoder, SoftMax as SoftMaxEncoder, Temporal as TemporalEncoder
from .functional import forward_heaviside, backward_rectangular, backward_polynomial, backward_sigmoid, backward_gaussian
from .layer import SRM0Linear, STDPLinear, MaxPool1d, MaxPool2d, MaxPool3d, AvgPool1d, AvgPool2d, AvgPool3d, Flatten, Unflatten
from .skeleton import Module
from .soma import Soma, IF, LIF, QIF, ExpIF, Izhikevich, KLIF, LIAF
from .surrogate import Rectangular, Polynomial, Sigmoid, Gaussian
from .synapse import Linear, Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm
try:
    from rich import print
except:
    pass