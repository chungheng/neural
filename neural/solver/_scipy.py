"""
Scipy Backend For Model
"""
import typing as tp
import numpy as np
from numpy import typing as npt
from numpy.lib import recfunctions as rfn
from scipy.integrate import OdeSolver, RK45, RK23, DOP853, Radau, LSODA, OdeSolution
from .basesolver import BaseSolver, Euler
from .. import types as tpe
from .. import errors as err
from ..backend import (
    BackendMixin,
    NumbaCPUBackendMixin,
)


class SciPySolver(BaseSolver):
    """SciPy OdeSolvers

    .. warning::

        Because this solver uses :py:module:`scipy.integrate.OdeSolver`
        instances to perform numerical integration, we need to transform
        neural's model states into the compatible shapes (1D array)
        as required by :code:`OdeSolver`s. As such, we cannot guarantee
        that the data types are perserved during integration step.
    """

    Supported_Backends = (BackendMixin, NumbaCPUBackendMixin)
    SolverCls: OdeSolver = OdeSolver

    def __init__(self, model: tpe.Model, **solver_options) -> None:
        if "vectorized" not in solver_options:
            solver_options["vectorized"] = False
        if "t_bound" not in solver_options:
            solver_options["t_bound"] = np.inf
        super().__init__(model, **solver_options)

        self.ode = self.get_wrapped_ode()
        self._t = 0
        self.set_initial_value(
            t0=self._t,
            **{
                name: self.model.initial_states[name][0].item()
                for name in self.model.initial_states.dtype.names
            }
        )
        self._dense_output = None
        # self.jac = self.model.jacobian

    def set_initial_value(self, t0: float = 0, **initial_states) -> None:
        """Change initial value of solver

        .. note::

            Since there is no unified API for resetting initial conditions
            for :py:module:`OdeSolver`, we just create a new instance of
            the solver.
        """
        model_initial_states = {
            name: self.model.initial_states[name][0].item()
            for name in self.model.initial_states.dtype.names
        }

        self._t = t0

        initial_states = {**model_initial_states, **model_initial_states}
        y0 = []
        for state_name in self.model.states.dtype.names:
            val = initial_states[state_name]
            if (np.isscalar(val) or val.size == 1):
                val = np.repeat(val, self.model.num)
            else:
                assert len(val) == self.model.num, (
                    f"Initial Value for state {state_name} has length {len(val)} "
                    f"but model number is {self.model.num}, cannot broadcast."
                )
            y0.append(val)
        y0 = np.vstack(val).ravel()
        if not self.model.Derivates:  # no gradients, no need for solver:
            self._solver = None
        else:
            self._solver = self.SolverCls(self.ode, t0, y0, **self.solver_options)

    def get_wrapped_ode(self) -> tp.Callable:
        """Return a callable ode function in the scipy.solve_ivp API"""
        def wrapped_ode(t, y, **input_args):
            self.model.states[:] = rfn.unstructured_to_structured(
                y.reshape((-1, self.model.num)).T, dtype=self.model.states.dtype
            )
            self.model.ode(**input_args)
            return rfn.structured_to_unstructured(self.model.gstates).T.ravel() * self.model.Time_Scale
        return wrapped_ode

    def step(self, d_t: float, **input_args) -> None:
        # If the model does not actually define any derivatives,
        # use Euler as default implementation
        if not self.model.Derivates:
            self._t += d_t
            Euler.step(self, d_t, **input_args)
            return

        # if the current time step is still within the dense_output interpolation
        # range, simply return the dense_output evaluated value
        if self._dense_output is not None and self._dense_output.t_max >= self._t + d_t:
            self._t += d_t
            return self._dense_output(self._t)

        # If new dense_output is required, recreate one
        interpolants = []
        ts = [self._solver.t]
        while np.abs(self._solver.t - self._t) < d_t: # must step at least by d_t amount
            msg = self._solver.step()
            if self._solver.status == "failed":
                raise err.NeuralSolverError(f"Solver failed with message: {msg}")
            sol = self._solver.dense_output()
            interpolants.append(sol)
            ts.append(self._solver.t)
        self._dense_output = OdeSolution(ts, interpolants)
        self._t += d_t

        if self._solver.t == self._t:
            self.model.states[:] = rfn.unstructured_to_structured(
                self._solver.y.reshape((-1, self.model.num)).T,
                dtype=self.model.states.dtype
            )
        else:
            self.model.states[:] = rfn.unstructured_to_structured(
                self._dense_output(self._t).reshape((-1, self.model.num)).T,
                dtype=self.model.states.dtype
            )


class RK45Solver(SciPySolver):
    SolverCls = RK45


class RK23Solver(SciPySolver):
    SolverCls = RK23


class DOP853Solver(SciPySolver):
    SolverCls = DOP853


class RadauSolver(SciPySolver):
    SolverCls = Radau


class LSODASolver(SciPySolver):
    SolverCls = LSODA
