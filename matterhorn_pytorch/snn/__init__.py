# -*- coding: UTF-8 -*-
"""
此文件夹放置SNN的一些基本模块。
"""


from . import functional
from .container import Spatial, Temporal, Sequential
from .decoder import SumSpike as SumSpikeDecoder, AverageSpike as AvgSpikeDecoder, MinTime as MinTimeDecoder, AverageTime as AvgTimeDecoder
from .encoder import Direct as DirectEncoder, Poisson as PoissonEncoder, Temporal as TemporalEncoder
from .layer import SRM0Linear, STDPLinear, MaxPool1d, MaxPool2d, MaxPool3d, AvgPool1d, AvgPool2d, AvgPool3d, Flatten, Unflatten, Dropout, Dropout1d, Dropout2d, Dropout3d
from .skeleton import Module
from .soma import IF, LIF, QIF, ExpIF, Izhikevich, KLIF, LIAF
from .surrogate import Rectangular, Polynomial, Sigmoid, Gaussian
from .synapse import Linear, Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm, Neurotransmitter
try:
    from rich import print
except:
    pass