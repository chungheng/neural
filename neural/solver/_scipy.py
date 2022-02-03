"""
Scipy Backend For Model
"""
import numpy as np
from scipy.integrate import (
    OdeSolver, RK45, RK23, DOP853, 
    Radau, LSODA, OdeSolution
)
from .base_backend import BaseSolver
from .. import types as tpe
from .. import errors as err

class SciPySolver(BaseSolver):
    SolverCls: OdeSolver = None

    def __init__(self, model: tpe.Model, **solver_options) -> None:
        super().__init__(model, **solver_options)

        self._ode = self.get_wrapped_ode()
        self._t = 0
        self.set_initial_value(t0=self._t, **self.model.initial_states)
        # cache options in case reinstantiation is 
        # required for initial value setting
        self._solver_options = solver_options
        self._dense_output = None
        self.jac = self.model.jacobian

    def states_to_vec(self, states: np.ndarray = None):
        pass

    def vec_to_states(self, vec: np.ndarray):
        pass

    def set_initial_value(self, t0=0, **initial_states):
        """Change initial value of solver

        .. note::

            Since there is no unified API for resetting initial conditions
            for :py:module:`OdeSolver`, we just create a new instance of
            the solver.
        """
        self._t = t0
        y0 = initial_states
        self._solver = OdeSolver(self._ode, t0, y0, **self._solver_options)

    def get_wrapped_ode(self) -> None:
        def wrapped_ode(t, states, **input_args):
            self.model.states = states
            self.ode(**input_args)
            return self.model.gstates
        return wrapped_ode

    def step(self, d_t: float, **input_args) -> None:
        if self._dense_output is not None and self._dense_output.t_max > self._t + d_t:
            self._t += d_t
            return self._dense_output(self._t)
        interpolants = []
        ts = []
        while np.abs(self._solver.t - self._t) < d_t:
            msg = self._solver.step()
            sol = self._solver.dense_output()
            interpolants.append(sol)
            ts.append(self._solver.t)
        self._dense_output = OdeSolution(ts, interpolants)
        self._t += d_t
        self.model.states = self._dense_output(self._t)

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