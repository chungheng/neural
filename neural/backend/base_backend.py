"""
Backend Classes For Model
"""
import inspect
import weakref
from functools import wraps
import typing as tp
import numpy as np
from tqdm.auto import tqdm

from neural.basemodel import Model
from .. import errors as err

# class Backend:
# def set_solver(self, solver: str, **solver_options):
#     assert solver in self.Solvers, err.NeuralBackendError(
#         f"Solver '{solver}' not supported"
#     )
#     self.solver = self.Solvers[solver](**solver_options)

# @classmethod
# def with_solver(cls, name, SolverCls):
#     def decorator(func):
#         if cls.Solvers is None:
#             cls.Solvers = {name: SolverCls}
#         else:
#             cls.Solvers[name] = SolverCls
#         @wraps(func)
#         def wrapped(*args, **kwargs):
#             return func(*args, **kwargs)
#         return wrapped
#     return decorator


class BaseSolver:
    def __init__(self, model: Model, **solver_options) -> None:
        self.model = weakref.proxy(model)

    def clip(self, states: dict = None) -> None:
        """Clip the State Variables

        Clip the state variables in-place after calling the numerical solver.

        The state variables are usually bounded, for example, binding
        variables are bounded between 0 and 1. However, numerical sovlers
        might cause the value of state variables exceed its bounds. A hard
        clip is forced here to ensure the state variables remain in the
        given bounds.
        """
        states = self.model.states if states is None else states
        for var, (lb, ub) in self.model.bounds.items():
            states[var].clip(lb, ub, out=states[var])

    def step(self, d_t: float, **input_args) -> None:
        """Step integration once"""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset Model

        Sets states to initial values, and sets gstates to 0.
        """
        for key, val in self.model.initial_states.items():
            if np.isscalar(val):
                self.model.states[key] = val
            else:
                self.model.states[key].fill(val)
        for key, val in self.model.gstates.items():
            if np.isscalar(val):
                self.model.gstates[key] = 0.0
            else:
                self.model.gstates[key].fill(0.0)


class Euler(BaseSolver):
    def step(self, d_t: float, **input_args) -> None:
        """Euler's method"""
        self.model.ode(**input_args)
        for var, grad in self.model.gstates.items():
            self.model.states[var] += d_t * grad
