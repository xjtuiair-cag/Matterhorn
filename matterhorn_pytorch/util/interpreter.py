# -*- coding: UTF-8 -*-
"""
（未完工）SNN一阶反应函数的表达式解析，可以将任意的一阶反应函数转换至可以被C++和CUDA计算的形式。
"""


import numpy as np
import torch
from typing import List, Tuple, Dict, Union, Any


COMMAND = (
    "NEG" , "ADD" , "SUB" , "MUL" , "DIV" , "NOT" , "AND" , "ORR" ,
    "XOR" , "EQ"  , "NE"  , "LT"  , "LE"  , "GT"  , "GE"  , "JMP" ,
    "POW" , "EXP" , "LOG" , "SIN" , "COS" , "TAN" , "COT" , "SEC" ,
    "CSC" , "ASIN", "ACOS", "ATAN", "ABS" , "SGN" , "RE"  , "IM"  ,
)
SYMBOLS = (
    "!"   , "+"   , "-"   , "*"   , "/"   , "~"   , "&"   , "|"   ,
    "@"   , "=="  , "!="  , "<"   , "<="  , ">"   , ">="  , "?"   ,
    "^"   , "exp" , "ln"  , "sin" , "cos" , "tan" , "cot" , "sec" ,
    "csc" , "asin", "acos", "atan", "abs" , "sgn" , "re"  , "im"  ,
)
OPERAND_TYPE = (
    1     , 2     , 2     , 2     , 2     , 1     , 2     , 2     ,
    2     , 2     , 2     , 2     , 2     , 2     , 2     , 2     ,
    2     , 1     , 1     , 1     , 1     , 1     , 1     , 1     ,
    1     , 1     , 1     , 1     , 1     , 1     , 1     , 1     ,
)
PRIORITY = (
    7     , 3     , 3     , 4     , 4     , 2     , 1     , 1     ,
    1     , 0     , 0     , 0     , 0     , 0     , 0     , 0     ,
    5     , 6     , 6     , 6     , 6     , 6     , 6     , 6     ,
    6     , 6     , 6     , 6     , 6     , 6     , 6     , 6     ,
)


class Node:
    def __init__(self, val = 0, left = None, right = None) -> None:
        self.val = val
        self.left = left
        self.right = right
    

    def __repr__(self) -> str:
        if self.left is None and self.right is None:
            if isinstance(self.val, str):
                return "$%s" % (self.val,)
            elif isinstance(self.val, int):
                return "$%d" % (self.val,)
            elif isinstance(self.val, float):
                return "$%.2f" % (self.val,)
        if OPERAND_TYPE[self.val] == 1:
            return "%s(%s)" % (SYMBOLS[self.val], str(self.left))
        if OPERAND_TYPE[self.val] == 2:
            return "(%s %s %s)" % (str(self.left), SYMBOLS[self.val], str(self.right))
        return ""


def translate_stacks(expr: str):
    bracket_level = 0
    expr_stacks = []
    expr_buffer = ""
    is_operator = lambda s: isinstance(s, str) and (s in SYMBOLS)
    def push_back(es: List, eb: str):
        if eb == "-" and (not len(es) or is_operator(es[-1])):
            eb = "!"
        if len(eb):
            es.append(eb)
            eb = ""
        return es, eb
    
    last_is_op = False
    for idx, s in enumerate(expr):
        s = expr[idx]
        if s == "(":
            if bracket_level == 0:
                expr_stacks, expr_buffer = push_back(expr_stacks, expr_buffer)
            bracket_level += 1
        if bracket_level > 0:
            expr_buffer += s
        else:
            is_op = is_operator(s)
            if (last_is_op != is_op) or is_op:
                expr_stacks, expr_buffer = push_back(expr_stacks, expr_buffer)
                last_is_op = is_op
            expr_buffer += s
        if s == ")":
            bracket_level -= 1
            if bracket_level == 0:
                expr_stacks, expr_buffer = push_back(expr_stacks, translate_stacks(expr_buffer[1:-1]))
    
    expr_stacks, expr_buffer = push_back(expr_stacks, expr_buffer)
    if bracket_level != 0:
        raise ValueError("You have unclosed brackets. Check your expression.")
    return expr_stacks


def translate_tree_node(node: Any):
    if isinstance(node, Node):
        return node
    if isinstance(node, List):
        return translate_tree(node)
    if node in SYMBOLS:
        raise ValueError("Unknown symbol %s." % (node,))
    return Node(node)


def translate_tree(stacks: List):
    priority_max = max(PRIORITY)
    for p in range(priority_max, -1, -1):
        current_priority_symbols = [symbol if PRIORITY[SYMBOLS.index(symbol)] == p else None for symbol in SYMBOLS]
        idx = 0
        while idx < len(stacks):
            item = stacks[idx]
            if item in current_priority_symbols:
                op = SYMBOLS.index(item)
                if OPERAND_TYPE[op] == 1:
                    assert idx < len(stacks) - 1, "Wrong operand for symbol %s." % (item)
                    operand1 = translate_tree_node(stacks[idx + 1])
                    front = stacks[:idx]
                    rear = stacks[idx + 2:] if idx + 2 < len(stacks) else []
                    stacks = front + [Node(op, operand1)] + rear
                    idx -= 0
                elif OPERAND_TYPE[op] == 2:
                    assert idx > 0 and idx < len(stacks) - 1, "Wrong operand for symbol %s." % (item)
                    operand1 = translate_tree_node(stacks[idx - 1])
                    operand2 = translate_tree_node(stacks[idx + 1])
                    front = stacks[:idx - 1] if idx - 2 >= 0 else []
                    rear = stacks[idx + 2:] if idx + 2 < len(stacks) else []
                    stacks = front + [Node(op, operand1, operand2)] + rear
                    idx -= 1
            idx += 1
    return translate_tree_node(stacks[0])


def translate_variable(var: str) -> float:
    try:
        return float(var)
    except:
        return var


def tree_traverse_update_params(expr_tree: Node, param_list: List = []) -> List:
    fixed_params = ("u", "h", "x")
    if expr_tree.left is not None:
        param_list = tree_traverse_update_params(expr_tree.left, param_list)
    if expr_tree.right is not None:
        param_list = tree_traverse_update_params(expr_tree.right, param_list)
    if expr_tree.left is None and expr_tree.right is None:
        val = translate_variable(expr_tree.val)
        if val in fixed_params:
            expr_tree.val = fixed_params.index(val)
        elif val in param_list:
            expr_tree.val = param_list.index(val) + 3
        else:
            param_list.append(val)
            expr_tree.val = len(param_list) + 2
    return param_list


def translate(expr: str):
    expr_stacks = translate_stacks(expr)
    expr_tree = translate_tree(expr_stacks)
    param_list = tree_traverse_update_params(expr_tree)
    return expr_tree, param_list


def param_init(param_list: List, param_dict: Dict = {}):
    for idx, p in enumerate(param_list):
        if not isinstance(p, float):
            assert p in param_dict, "Unknown parameter %s" % (p,)
            param_list[idx] = param_dict[p]
    return torch.tensor(param_list, dtype = torch.float)


if __name__ == "__main__":
    # a = translate("1+5*exp(x)+3")
    # a = translate("(1.0/tau_m)*(-a_0*(h-u_rest)*(h-u_c)+x)*sin(x)")
    expr_tree, param_list = translate("1.0/tau_m*(-(h-u_rest)+x)")
    print(expr_tree)
    print(param_list)