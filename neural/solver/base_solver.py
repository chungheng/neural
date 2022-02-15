"""
Backend Classes For Model
"""
from abc import abstractmethod
import inspect
import weakref
from functools import wraps
import typing as tp
import numpy as np
from tqdm.auto import tqdm

from .. import errors as err
from .. import types as tpe


class BaseSolver:
    def __init__(self, model: tpe.Model, **solver_options) -> None:
        self.model = weakref.proxy(model)

        # cache options in case reinstantiation is
        # required for initial value setting
        self.solver_options = solver_options or {}

    @abstractmethod
    def step(self, d_t: float, **input_args) -> None:
        """Step integration once"""


class Euler(BaseSolver):
    def step(self, d_t: float, **input_args) -> None:
        """Euler's method"""
        self.model.ode(**input_args)
        for var, grad in self.model.gstates.items():
            self.model.states[var] += d_t * self.model.Time_Scale * grad
