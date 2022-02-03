"""
SOLVERS for Model
"""
import typing as tp
import numpy as np
from scipy.integrate import solve_ivp
from .. import types as tpe

def _dict_iadd_(dct_a: dict, dct_b: dict) -> dict:
    """Add dictionaries inplace"""
    for key in dct_a.keys():
        dct_a[key] += dct_b[key]
    return dct_a


def _dict_add_(dct_a: dict, dct_b: dict, out: dict = None) -> dict:
    """Add dictionaries"""
    if out is None:
        out = dct_a.copy()
    else:
        for key, val in dct_a.items():
            out[key] = val
    _dict_iadd_(out, dct_b)
    return out


def _dict_add_scalar_(dct_a: dict, dct_b: dict, sal: float, out: dict = None) -> dict:
    """Add dictionaries with scaling"""
    if out is None:
        out = dct_a.copy()
    else:
        for key, val in dct_a.items():
            out[key] = val
    for key in dct_a.keys():
        out[key] += sal * dct_b[key]
    return out


def register_solver(*args) -> tp.Callable:
    """Decorator for registering solver"""
    # when there is no args
    if len(args) == 1 and callable(args[0]):
        args[0]._solver_names = [args[0].__name__]
        return args[0]
    else:

        def wrapper(func):
            func._solver_names = [func.__name__] + list(args)
            return func

        return wrapper


def ode_wrapper(model: tpe.Model, states: dict = None, **input_args) -> dict:
    """
    A wrapper for calling `ode` with arbitrary variable than `model.states`

    Arguments:
        states: state variables.
        gstates: gradient of state variables.
    """
    if states is not None:
        _states = model.states
        model.states = states

    model.ode(**input_args)

    if states is not None:
        model.states = _states

    return model.gstates


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


@register_solver("euler", "forward")
def forward_euler(model, d_t: float, **input_args) -> None:
    """
    Forward Euler method.

    Arguments:
        d_t (float): time steps.
    """
    model.ode(**input_args)

    for key in model.gstates:
        model.states[key] += d_t * model.gstates[key]
    model.clip()


@register_solver("mid")
def midpoint(model, d_t: float, **input_args) -> None:
    """
    Implicit Midpoint method.

    Arguments:
        d_t (float): time steps.
    """
    states = model.states.copy()

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


@register_solver("rk4")
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


@register_solver("scipy_ivp")
def scipy_ivp(model, d_t: float, t: np.ndarray = None, **input_args) -> None:
    """
    Wrapper for scipy.integrate.solve_ivp

    Arguments:
        d_t (float): time steps.
    """
    solver = input_args.pop("solver", model._scipy_solver)
    # ensure that the dictionary keys are matched
    keys = list(model.gstates.keys())
    vectorized_states = [model.states[k] for k in keys]

    def f(states, t):  # note that the arguments are dummies,
        # not used anywhere in the body of the function
        res = ode_wrapper(model, **input_args)
        return [res[k] for k in keys]

    res = solve_ivp(
        f, [0, d_t], vectorized_states, method=solver, t_eval=[d_t], events=None
    )
    if res is not None:
        model.states.update(
            {k: res.y[:, -1][n] for n, k in enumerate(keys)}
        )  # indexing res.y[:, -1] incase multiple steps are returned
    model.clip()
    # TODO: `events` can be used for spike detection
