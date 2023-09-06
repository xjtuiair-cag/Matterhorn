# -*- coding: UTF-8 -*-
"""
此文件夹放置SNN的一些成品模型。
"""

from .sew import ResADD, ResAND, ResIAND, SEWBlock, SEWRes18
try:
    from rich import print
except:
    pass