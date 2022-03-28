"""Solvers for Model
"""
import collections
from .basesolver import BaseSolver, Euler

# from ._numba import NumbaEulerSolver, NumbaMidpointSolver, NumbaSolver

SOLVERS = collections.namedtuple("Solvers", "euler")(
    Euler,
)
