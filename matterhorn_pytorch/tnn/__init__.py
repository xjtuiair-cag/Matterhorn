# -*- coding: UTF-8 -*-
"""
TNN（Temporal Neural Network）模块。

提供脉冲簇编解码、兴奋/抑制柱、认知柱、STDP 与训练工具，
用于基于首次脉冲时间戳的时空计算与反向传播训练。
参考文献：Smith J E. Space-Time Computing with Temporal Neural Networks.
"""


from . import functional
from .volley import (
    EMPTY_SPIKE,
    is_spike_present,
    normalize_time_volley,
    denormalize_time_volley,
    value_to_time_volley,
    time_to_value_volley,
    merge_time_volleys,
    onoff_encode,
    OnOffEncoder
)
from .response import compound_response, membrane_potential
from .neuron import ExcitatoryNeuron
from .inhibition import (
    feedforward_inhibit,
    lateral_inhibit,
    FeedforwardInhibition,
    LateralInhibition
)
from .column import ExcitatoryColumn, build_random_connectivity
from .cognitive import CognitiveColumn
from .stdp import stdp_gate, volley_stdp_update, mean_time_to_weight
from .container import Sequential
from .training import (
    spike_time_mse_loss,
    train_excitatory_neuron_stdp,
    train_column_mean_weights,
    TNNTrainer
)
from .component import Bitonic, Firing
