"""Solvers for Model
"""
import collections
from .basesolver import BaseSolver, Euler
from ._scipy import (
    SciPySolver,
    RK45Solver,
    RK23Solver,
    DOP853Solver,
    RadauSolver,
    LSODASolver,
)
# from ._numba import NumbaEulerSolver, NumbaMidpointSolver, NumbaSolver

SOLVERS = collections.namedtuple(
    "AvailableSolvers",
    [
        "euler",
        "rk45",
        "rk23",
        "dop853",
        "radau",
        "lsoda"
    ])(
        euler = Euler,
        rk45 = RK45Solver,
        rk23 = RK23Solver,
        dop853 = DOP853Solver,
        radau = RadauSolver,
        lsoda = LSODASolver,
    )