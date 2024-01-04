# -*- coding: UTF-8 -*-
"""
此文件夹放置TNN的一些基本模块。
"""


from .functional import t_add, t_min, t_xmin, t_max, t_xmax, t_eq, t_ne, t_lt, t_le, t_gt, t_ge, s_add, s_min, s_xmin, s_max, s_xmax, s_eq, s_ne, s_lt, s_le, s_gt, s_ge
try:
    from rich import print
except:
    pass