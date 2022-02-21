"""
Backend Classes For Model
"""
from abc import abstractmethod
from warnings import warn
import weakref
import typing as tp
from ..backend import BackendMixin
from .. import errors as err
from .. import types as tpe


class BaseSolver:
    Supported_Backends: tp.Iterable[BackendMixin] = None

    def __init__(self, model: tpe.Model, **solver_options) -> None:
        if self.Supported_Backends is not None and (
            model.backend is not None and model.backed not in self.Supported_Backends
        ):
            warn(
                f"Model backend '{model.backend}' is not supported by this solver {self.__class__}",
                err.NeuralSolverWarning,
            )
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