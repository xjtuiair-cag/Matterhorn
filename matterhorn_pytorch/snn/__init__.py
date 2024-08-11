# -*- coding: UTF-8 -*-
"""
此文件夹放置SNN的一些基本模块。
"""


from . import functional
from .container import Spatial, Temporal, Sequential, ModuleList, ModuleDict, Agent
from .decoder import SumSpike as SumSpikeDecoder, AverageSpike as AvgSpikeDecoder, MinTime as MinTimeDecoder, AverageTime as AvgTimeDecoder
from .encoder import Direct as DirectEncoder, Analog as AnalogEncoder, Poisson as PoissonEncoder, Temporal as TemporalEncoder
from .layer import STDPLinear, STDPConv2d, MaxPool1d, MaxPool2d, MaxPool3d, AvgPool1d, AvgPool2d, AvgPool3d, MaxUnpool1d, MaxUnpool2d, MaxUnpool3d, Upsample, Flatten, Unflatten, Dropout, Dropout1d, Dropout2d, Dropout3d
from .skeleton import Module
from .soma import IF, LIF, QIF, ExpIF, Izhikevich, KLIF, LIAF
from .firing import Rectangular, Polynomial, Sigmoid, Gaussian, Floor, Ceil, Round
from .synapse import Linear, WSLinear, Conv1d, WSConv1d, Conv2d, WSConv2d, Conv3d, WSConv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm, Identity