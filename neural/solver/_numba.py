"""
Numba CPU Solvers
"""
import numba
import typing as tp
import numpy as np
from .. import types as tpe
from .basesolver import BaseSolver
from ..backend import NumbaCPUBackendMixin

class NumbaSolver(BaseSolver):
    Supported_Backends = (NumbaCPUBackendMixin,)

    @numba.njit
    def _step(self, d_t: float, **input_args) -> None:
        raise NotImplementedError

    def step(self, d_t: float, **input_args) -> None:
        """Perform one integration step of size d_t"""
        return self._step(d_t, **input_args)

class RungeKutta(NumbaSolver):
    """Base class for explicit Runge-Kutta methods.

    Attributes:
        A: ndarray, shape (n_stages, n_stages)
            Coefficients for combining previous RK stages to compute the next
            stage. For explicit methods the coefficients at and above the main
            diagonal are zeros.
        B: ndarray, shape (n_stages,)
            Coefficients for combining RK stages for computing the final
            prediction.
        C: ndarray, shape (n_stages,)
            Coefficients for incrementing time for consecutive RK stages.
            The value for the first stage is always zero.
        K: ndarray, shape (n_stages + 1, num)
            Storage array for putting RK stages here. Stages are stored in rows.
            The last row is a linear combination of the previous rows with
            coefficients
    """
    A: np.ndarray = NotImplemented
    B: np.ndarray = NotImplemented
    C: np.ndarray = NotImplemented

    def __init__(self, model: tpe.Model, **solver_options) -> None:
        super().__init__(model=model, **solver_options)
        self.n_stages = len(self.A)
        assert len(self.A) == len(self.B) == len(self.C)
        self.K = np.recarray((self.n_stages, self.model.num), dtype=self.model.gstates.dtype)

    @numba.njit
    def _step(self, d_t: float, **input_args) -> None:
        self.K[0] = self.model.gstates
        curr_states = self.model.states.copy()
        for s, (a, c) in enumerate(zip(self.A[1:], self.C[1:]), start=1):
            gstates = self.K[:s].T @ a[:s]
            self.model.states = curr_states + d_t * self.model.Time_Scale * gstates
            self.model.ode(**input_args)
            self.K[s] = self.model.gstates

        self.model.states = curr_states + d_t * self.model.Time_Scale * (self.K[:-1].T @ self.B)
        self.model.ode(**input_args)
        self.K[-1] = self.model.gstates

class NumbaRK23Solver(RungeKutta):
    C = np.array([0, 1/2, 3/4])
    A = np.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [0, 3/4, 0]
    ])
    B = np.array([2/9, 1/3, 4/9])

class NumbaRK45Solver(RungeKutta):
    C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A = np.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])

class NumbaEulerSolver(NumbaSolver):
    @numba.njit
    def _step(self, d_t: float, **input_args) -> None:
        self.model.ode(**input_args)
        for var in self.model.Derivates:
            self.model.states[var] += d_t * self.model.Time_Scale * self.model.gstates[var]