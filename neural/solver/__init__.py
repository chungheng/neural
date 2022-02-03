"""Solvers for Model
"""
import collections
from .base_solver import BaseSolver, Euler
from ._scipy import SciPySolver

SOLVERS = collections.namedtuple(
    "Solvers", "euler scipy"
)(Euler, SciPySolver)
