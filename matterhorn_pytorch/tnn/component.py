# -*- coding: UTF-8 -*-
"""
TNN的组件，构建TNN兴奋柱和抑制柱的基本组件。
"""


import torch
import torch.nn as nn
from typing import Any
import matterhorn_pytorch.tnn.functional as F
try:
    from rich import print
except:
    pass