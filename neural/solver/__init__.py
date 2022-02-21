"""Solvers for Model
"""
import collections
from .basesolver import BaseSolver, Euler
from ._scipy import (
    RK45Solver,
    RK23Solver,
    DOP853Solver,
    RadauSolver,
    LSODASolver,
)

SOLVERS = collections.namedtuple("Solvers", "euler RK45 RK23 DOP853 Radau LSODA")(
    Euler,
    RK45Solver,
    RK23Solver,
    DOP853Solver,
    RadauSolver,
    LSODASolver,
)
