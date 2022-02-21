"""
Numba CPU Solvers
"""
import numba
import typing as tp
import numpy as np
from scipy.integrate import solve_ivp
from .. import types as tpe
from .basesolver import BaseSolver
from ..backend import NumbaCPUBackendMixin, NumbaCUDABackendMixin


class NumbaSolver(BaseSolver):
    Supported_Backends = (NumbaCPUBackendMixin,)

    @numba.njit
    def _step(self, d_t: float, **input_args) -> None:
        pass

    def step(self, d_t: float, **input_args) -> None:
        """Euler's method"""
        return self._step(d_t, **input_args)


class NumbaEulerSolver(NumbaSolver):
    @numba.njit
    def _step(self, d_t: float, **input_args) -> None:
        self.model.ode(**input_args)
        for var, grad in self.model.gstates.items():
            self.model.states[var] += d_t * self.model.Time_Scale * grad


# def _dict_iadd_(dct_a: dict, dct_b: dict) -> dict:
#     """Add dictionaries inplace"""
#     for key in dct_a.keys():
#         dct_a[key] += dct_b[key]
#     return dct_a


# def _dict_add_(dct_a: dict, dct_b: dict, out: dict = None) -> dict:
#     """Add dictionaries"""
#     if out is None:
#         out = dct_a.copy()
#     else:
#         for key, val in dct_a.items():
#             out[key] = val
#     _dict_iadd_(out, dct_b)
#     return out


# def _dict_add_scalar_(dct_a: dict, dct_b: dict, sal: float, out: dict = None) -> dict:
#     """Add dictionaries with scaling"""
#     if out is None:
#         out = dct_a.copy()
#     else:
#         for key, val in dct_a.items():
#             out[key] = val
#     for key in dct_a.keys():
#         out[key] += sal * dct_b[key]
#     return out


def increment(
    model: tpe.Model, d_t: float, states: dict, out_states: dict = None, **input_args
) -> dict:
    """
    Compute the increment of state variables.

    This function is used for advanced numerical methods.

    Arguments:
        d_t: time steps
        states: state variables
        out_states: output states

    Keyword Arguments:
        Arguments for :py:func:`model.ode`.

    Returns:
        A dictionary that contains
    """
    if out_states is None:
        out_states = states.copy()

    gstates = ode_wrapper(model, states, **input_args)

    for key in gstates:
        out_states[key] = d_t * gstates[key]

    return out_states


def _increment(
    model: tpe.Model, d_t: float, states: dict, out_states: dict = None, **input_args
) -> dict:
    """
    Compute the increment of state variables.

    This function is used for advanced numerical methods.

    Arguments:
        d_t: time steps
        states: state variables
        out_states: output states

    Keyword Arguments:
        Arguments for :py:func:`model.ode`.

    Returns:
        A dictionary that contains
    """
    if out_states is None:
        out_states = states.copy()

    gstates = ode_wrapper(model, states, **input_args)

    for key in gstates:
        out_states[key] = d_t * gstates[key]

    return out_states


def _forward_euler(
    model, d_t: float, states: dict, out_states: dict = None, **input_args
) -> dict:
    """
    Forward Euler method with arbitrary variable than `model.states`.

    This function is used for advanced numerical methods.

    Arguments:
        d_t (float): time steps.
        states (dict): state variables.
    """
    if out_states is None:
        out_states = states.copy()

    increment(d_t, states, out_states, **input_args)

    for key in out_states:
        out_states[key] += states[key]

    model.clip(out_states)
    return out_states


class NumbaMidpointSolver(NumbaSolver):
    @numba.njit
    def _step(self, d_t: float, **input_args) -> None:
        states = self.model.states.copy()  # freeze state

        _forward_euler(0.5 * d_t, model.states, states, **input_args)
        _forward_euler(d_t, states, model.states, **input_args)


@register_solver("heun")
def heun(model, d_t: float, **input_args) -> None:
    """
    Heun's method.

    Arguments:
        d_t (float): time steps.
    """
    incr1 = increment(d_t, model.states, **input_args)
    tmp = _dict_add_(model.states, incr1)
    incr2 = increment(d_t, tmp, **input_args)

    for key in model.states.keys():
        model.states[key] += 0.5 * incr1[key] + 0.5 * incr2[key]
    model.clip()


def runge_kutta_4(model, d_t: float, **input_args) -> None:
    """
    Runge Kutta method.

    Arguments:
        d_t (float): time steps.
    """
    tmp = model.states.copy()
    k1 = increment(d_t, model.states, **input_args)

    _dict_add_scalar_(model.states, k1, 0.5, out=tmp)
    model.clip(tmp)
    k2 = increment(d_t, tmp, **input_args)

    _dict_add_scalar_(model.states, k2, 0.5, out=tmp)
    model.clip(tmp)
    k3 = increment(d_t, tmp, **input_args)

    _dict_add_(model.states, k3, out=tmp)
    model.clip(tmp)
    k4 = increment(d_t, tmp, **input_args)

    for key in model.states.keys():
        incr = (k1[key] + 2.0 * k2[key] + 2.0 * k3[key] + k4[key]) / 6.0
        model.states[key] += incr
    model.clip()
