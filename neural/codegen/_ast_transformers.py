"""Transformers for AST Nodes
"""
import math
from numbers import Number
import typing
import ast
import textwrap
import inspect
import random
import numpy
import sympy
import sympy.stats
import sys

if sys.version_info < (3, 9):
    from astunparse import unparse
else:
    from ast import unparse
from .. import errors as err
from .ufuncs import _numpy as NUMPY_FUNCTIONS
from .ufuncs import _random as RANDOM_FUNCTIONS
from .ufuncs import _math as MATH_FUNCTIONS


def _get_function_module(func: typing.Callable) -> str:
    """Get the name of the module"""
    if inspect.getmodule(func) is not None:
        return inspect.getmodule(func)
    if isinstance(func, numpy.ufunc):
        return numpy
    return None


class NumPy2SymPy(ast.NodeTransformer):
    """Convert function calls to SymPy calls

    For Calls:

        1. Builtin function calls
        2. NumPy calls that exist in SymPy's default namespace
        3. `random` calls that exist in `sympy.stats` module.

    For IfExp:
        Convert IfExp to sympy.Piecewise

    Arguments:

    """
    _rng_ctr = 0
    _globals = dict()

    def __init__(self, global_dct=None):
        if global_dct is not None:
            self._globals.update(global_dct)

    def visit_Assign(self, node: ast.Assign) -> typing.Any:
        """Convert Tuple assignment to multiple single assignments

        a = 1
        a,b = x
        a = b = 1
        a,b = 1, 0
        a = a,b = x, y
        """
        self.generic_visit(node)
        if len(node.targets) == 1 and not isinstance(node.targets[0], ast.Tuple):
            return node
        nodes = []
        for target in node.targets:
            if isinstance(target, ast.Tuple):
                if isinstance(node.value, ast.Tuple):  # a,b = 0,1
                    if len(target.elts) != len(node.value.elts):
                        raise err.NeuralCodeGenError(
                            f"Number of targets is different from number of values in assingment: {unparse(node)}"
                        )
                    nodes += [
                        ast.Assign(targets=[_tgt], value=_val)
                        for _tgt, _val in zip(target.elts, node.value.elts)
                    ]
                else:
                    val = unparse(node.value)
                    nodes += [
                        ast.parse(f"{unparse(_tgt)} = {val}[{n}]").body[0]
                        for n, _tgt in enumerate(target.elts)
                    ]
            else:
                nodes += [ast.Assign(targets=[target], value=node.value)]
        return nodes

    def visit_Compare(self, node):
        self.generic_visit(node)
        return ast.Call(
            func=ast.parse('sympy.Piecewise').body[0].value,
            args=[
                ast.Tuple(elts=[ast.Constant(value=1), node]),
                ast.Tuple(
                    elts=[
                        ast.Constant(value=0),
                        ast.Constant(value=True),
                    ]
                ),
            ],
            keywords=[],
        )

    def visit_Constant(self, node):
        """Convert constant numeric value to unevaluated expressions in sympy to prevent simplification"""
        self.generic_visit(node)
        if isinstance(node.value, Number):
            return ast.Call(
                func=ast.parse('sympy.UnevaluatedExpr').body[0].value,
                args=[node],
                keywords=[],
            )
        return node

    def visit_Call(self, node):
        """Handle Call Node

        For each Call, we:
        1. determine the module of the function being called
        2. find the sympy equivalent function if necessary
        """
        self.generic_visit(node)
        try:
            func = eval(unparse(node.func), self._globals)
        except Exception as e:
            raise err.NeuralCodeGenError(
                f"Function '{unparse(node.func)}' not understood"
            ) from e
        func_module = _get_function_module(func)

        if func_module == numpy:
            if func not in NUMPY_FUNCTIONS.MAPPING:
                raise err.NeuralCodeGenError(
                    f"Function {func} not supported, "
                    f"use one of {NUMPY_FUNCTIONS.SUPPORTED}"
                )
            node.func = ast.parse(NUMPY_FUNCTIONS.MAPPING[func]).body[0].value
            return node
        if func_module == math:
            if func not in MATH_FUNCTIONS.MAPPING:
                raise err.NeuralCodeGenError((
                    f"Function {func} not supported, "
                    f"use one of {MATH_FUNCTIONS.SUPPORTED}"
                ))
            node.func = ast.parse(MATH_FUNCTIONS.MAPPING[func]).body[0].value
            return node
        if func_module == random:  # random number generators
            if func not in RANDOM_FUNCTIONS.MAPPING:
                raise err.NeuralCodeGenError((
                    f"Function {func} not supported, "
                    f"use one of {RANDOM_FUNCTIONS.SUPPORTED}"
                ))
            _random_args = RANDOM_FUNCTIONS.MAPPING[func]["random_params"]
            _sympy_args = RANDOM_FUNCTIONS.MAPPING[func]["sympy_params"]
            for kwarg in node.keywords:
                _arg_idx = list(_random_args.keys()).index(kwarg.arg)
                kwarg.arg = list(_sympy_args.keys())[_arg_idx]
            node.args = [ast.Constant(value=f"rng_{self._rng_ctr}")] + node.args
            self._rng_ctr += 1
            node.func = ast.parse(RANDOM_FUNCTIONS.MAPPING[func]["func"]).body[0].value
            return node
        # do not change node if the function calls a builtin method.
        if inspect.isbuiltin(func):
            return node

        # Other functions
        raise NotImplementedError(f"Function '{unparse(node)}' not understood.")

    def visit_If(self, node):
        raise NotImplementedError(
            (
                "If/Else clauses detected. For conditional assignment to variables, "
                "please use 'x = a if cond1 else b if cond2 else c' syntax instead. "
                f"\nSource Code:\n{textwrap.indent(unparse(node), prefix=' '*4)}"
            )
        )

    def visit_IfExp(self, node):
        """Handle RHS of conditional assignments"""
        if not isinstance(node.parent, ast.IfExp):
            self._ifexp_cases = []
        self._ifexp_cases.append(ast.Tuple(elts=[node.body, node.test], ctx=ast.Load()))
        if isinstance(node.orelse, ast.IfExp):
            self.visit_IfExp(node.orelse)
        else:
            self._ifexp_cases.append(
                ast.Tuple(elts=[node.orelse, ast.Constant(value=True)], ctx=ast.Load())
            )
        return ast.Call(
            func=ast.parse('sympy.Piecewise').body[0].value,
            args=self._ifexp_cases,
            keywords=[],
        )
