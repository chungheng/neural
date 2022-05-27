import weakref
import numpy as np
from scipy.integrate._ivp.common import warn_extraneous
from .basesolver import BaseSolver
from .. import types as tpe


class RungeKutta(BaseSolver):
    """Base class for explicit Runge-Kutta methods."""

    A: np.ndarray = NotImplemented
    B: np.ndarray = NotImplemented
    n_stages: int = NotImplemented

    def __init__(self, model: tpe.Model, **solver_options):
        warn_extraneous(solver_options)
        super().__init__(model, **solver_options)
        self.model = weakref.proxy(model)
        self.gstates_cache = {
            var: np.zeros(
                (self.n_stages + 1,) + self.model.gstates[var].shape,
                dtype=self.model.gstates[var].dtype,
            )
            for var in self.model.gstates.dtype.names
        }

    def step(self, d_t: float, **input_args) -> None:
        """Perform a+ single Runge-Kutta step.

        This function computes a prediction of an explicit Runge-Kutta method and
        also estimates the error of a less accurate method.

        Parameters:
            d_t : Step size to use.

        Keyword Arguments:
            input_args: input arguments for :code:`Model.ode`
        """
        curr_states = self.model.states.copy()
        for var in self.model.gstates.dtype.names:
            self.gstates_cache[var][0] = self.model.gstates[var]
        for s, a in enumerate(self.A[1:], start=1):
            for var, grad in self.gstates_cache.items():
                dy = (grad[:s].T @ a[:s]) * d_t * self.model.Time_Scale
                self.model.states[var] = curr_states[var] + dy
            self.model.ode(**input_args)
            for var, grad in self.gstates_cache.items():
                self.gstates_cache[var][s] = self.model.gstates[var]
        for var in curr_states.dtype.names:
            val = curr_states[var]
            if var in self.gstates_cache:
                self.model.states[var] = val + d_t * self.model.Time_Scale * (
                    self.gstates_cache[var][:-1].T @ self.B
                )
            else:
                self.model.states[var] = val
        self.model.ode(**input_args)
        for var in self.model.gstates.dtype.names:
            self.gstates_cache[var][-1] = self.model.gstates[var]


class RK23(RungeKutta):
    """Explicit Runge-Kutta method of order 3(2)."""

    n_stages = 3
    A = np.array([[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]])
    B = np.array([2 / 9, 1 / 3, 4 / 9])


class RK45(RungeKutta):
    """Explicit Runge-Kutta method of order 5(4)."""

    n_stages = 6
    A = np.array(
        [
            [0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        ]
    )
    B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
