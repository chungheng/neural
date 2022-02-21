"""
Scipy Backend For Model
"""
import typing as tp
import numpy as np
from scipy.integrate import OdeSolver, RK45, RK23, DOP853, Radau, LSODA, OdeSolution
from .basesolver import BaseSolver, Euler
from .. import types as tpe
from .. import errors as err
from ..backend import (
    BackendMixin,
    NumbaCPUBackendMixin,
    NumbaCUDABackendMixin,
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
        self.set_initial_value(t0=self._t, **self.model.initial_states)
        self._dense_output = None
        self.jac = self.model.jacobian

    def set_initial_value(self, t0: float = 0, **initial_states):
        """Change initial value of solver

        .. note::

            Since there is no unified API for resetting initial conditions
            for :py:module:`OdeSolver`, we just create a new instance of
            the solver.
        """
        self._t = t0
        y0 = self._states_to_vec(
            {
                var: np.repeat(val, self.model.num)
                if (np.isscalar(val) or val.size == 1)
                else val
                for var, val in {**self.model.initial_states, **initial_states}.items()
            }
        )
        if not self.model.Derivates:  # no gradients, no need for solver:
            self._solver = None
        else:
            self._solver = self.SolverCls(self.ode, t0, y0, **self.solver_options)

    def _states_to_vec(self, states: dict) -> np.ndarray:
        return np.vstack(list(states.values())).ravel()

    def _vec_to_states(self, vec: np.ndarray, ref_state: dict = None) -> dict:
        return {
            var: arr
            for var, arr in zip(
                (
                    ref_state.keys()
                    if ref_state is not None
                    else self.model.states.keys()
                ),
                vec.reshape((-1, self.model.num)),
            )
        }

    def get_wrapped_ode(self) -> tp.Callable:
        def wrapped_ode(t, y, **input_args):
            self.model.states.update(self._vec_to_states(y, self.model.gstates))
            self.model.ode(**input_args)
            return self._states_to_vec(self.model.gstates) * self.model.Time_Scale

        return wrapped_ode

    def step(self, d_t: float, **input_args) -> None:
        if not self.model.Derivates:
            self._t += d_t
            Euler.step(self, d_t, **input_args)
            return
        if self._dense_output is not None and self._dense_output.t_max >= self._t + d_t:
            self._t += d_t
            return self._dense_output(self._t)
        interpolants = []
        ts = [self._solver.t]
        while np.abs(self._solver.t - self._t) < d_t:
            msg = self._solver.step()
            sol = self._solver.dense_output()
            interpolants.append(sol)
            ts.append(self._solver.t)
        self._dense_output = OdeSolution(ts, interpolants)
        self._t += d_t
        self.model.states.update(
            self._vec_to_states(
                self._solver.y
                if self._solver.t == self._t
                else self._dense_output(self._t),
                self.model.gstates,
            )
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
