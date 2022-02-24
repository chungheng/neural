# pylint:disable=method-hidden
"""
Base model class for neurons and synapses.
"""
from functools import cache
import inspect
import copy
import ast
import textwrap
from abc import abstractmethod
from warnings import warn
from inspect import getfullargspec
import typing as tp
import numpy as np
from tqdm.auto import tqdm
from .solver import BaseSolver, Euler
from . import errors as err
from .utils.array import (
    cudaarray_to_cpu,
    get_array_module,
)
from .backend import BackendMixin


class FindDerivates(ast.NodeVisitor):
    def __init__(self):
        self.Derivates = []

    def visit_Assign(self, node: ast.Assign) -> tp.Any:
        for t in node.targets:
            if (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
                and t.attr.startswith("d_")
            ):
                self.Derivates.append(t.attr[2:])


class Model:
    """Base Model Class"""

    Default_States: dict = None
    Default_Params: dict = None
    Time_Scale: float = 1.0
    solver: BaseSolver = Euler
    backend: BackendMixin = None

    @classmethod
    @property
    @cache
    def bounds(cls) -> dict:
        """(upperbound, lowerbound) of each state variable, if specified"""
        dct = {}
        for key, val in cls.Default_States.items():
            if not np.isscalar(val):
                if len(val) != 3:
                    raise err.NeuralModelError(
                        f"Variable {key} should be a scalar of a iterable "
                        "of 3 elements (initial value, upper bound, lower bound) "
                        f"but {val} is given."
                    )
                dct[key] = np.asarray(val[1:])
        return dct

    @classmethod
    @property
    @cache
    def Variables(cls) -> dict:
        """A dictionary mapping variable name to physical type"""
        return {
            **{var: "states" for var in cls.Default_States.keys()},
            **{var: "gstates" for var in cls.Derivates},
            **{var: "params" for var in cls.Default_Params.keys()},
        }

    @classmethod
    @property
    @cache
    def Derivates(cls) -> tuple:
        """Return all state variables whose gradients are defined in ode()"""
        vis = FindDerivates()
        vis.visit(ast.parse(textwrap.dedent(inspect.getsource(cls.ode))))
        return tuple([var for var in vis.Derivates if var in cls.Default_States])

    @classmethod
    @property
    @cache
    def Inputs(cls) -> dict:
        """Input variables and their default values for ode() and post()"""
        # parse inputs
        inputs = dict()
        for func in [cls.ode, cls.post]:
            argspec = getfullargspec(func)
            if argspec.defaults is None:
                continue
            if argspec.varargs is not None:
                raise err.NeuralModelError(
                    f"Variable positional argument is not allowed in {func}"
                )
            if getattr(argspec, "varkw", None) is not None:
                raise err.NeuralModelError(
                    f"Variable keyword argument is not allowed in {func}"
                )
            for val, key in zip(argspec.defaults[::-1], argspec.args[::-1]):
                dtype = val.dtype if isinstance(val, np.generic) else np.float_
                inputs[key] = dtype(val)
        return inputs

    def __init__(
        self,
        num: int = 1,
        callback: tp.Union[tp.Callable, tp.Iterable[tp.Callable]] = None,
        backend: BackendMixin = BackendMixin,
        backend_kws: dict = None,
        solver: BaseSolver = Euler,
        solver_kws: dict = None,
        **kwargs,
    ):
        """
        Arguments:
            num: number of components for the model
            callback: iterable of callback functions.
              callback functions should have model instance as first and
              only argument.
            backend: which backend class to use for the model
            backend_kws: keyword arguments for initializing the backend
            solver: solver class to use for the model
            solver_kws: keyword arguments for initializing the solver

        Keyword Arguments:
            Are assumed to be values for states or parameters
        """
        dtype = []
        for key, val in self.Default_Params.items():
            _dtype = val.dtype.type if isinstance(val, np.generic) else np.float_
            dtype.append((key, _dtype))
        for key, val in self.Default_States.items():
            assert np.isscalar(val) or len(val) == 3, err.NeuralModelError(
                f"Variable {key} should be a scalar of a iterable "
                "of 3 elements (initial value, upper bound, lower bound) "
                f"but {val} is given."
            )
            _state = val if np.isscalar(val) else val[0]
            _dtype = _state.dtype.type if isinstance(_state, np.generic) else np.float_
            dtype.append((key, _dtype))
            if key in self.Derivates:
                dtype.append((f"d_{key}", _dtype))
        self.dtype = np.dtype(dtype)

        # _data is a structured array containing all the variable data
        self._data = np.zeros(num, dtype=self.dtype)

        # states, params, gstates are slices of the array
        # created in recast()
        self.recast()

        # populate params/states values
        for key, val in self.Default_Params.items():
            self._data[key][:] = val

        # save initial states
        for key, val in self.Default_States.items():
            _state = val if np.isscalar(val) else val[0]
            self._data[key][:] = _state
        self.initial_states = copy.deepcopy(self.states[0])

        # set additional variables
        for key, val in kwargs.items():
            if key not in self.dtype.names:
                raise err.NeuralModelError(f"Unexpected variable '{key}'")
            self._data[key][:] = val

        # update initial states
        self.initial_states = copy.deepcopy(self.states)
        self.callbacks = []

        callback = [] if callback is None else callback
        self.add_callback(callback)

        backend_kws = backend_kws or {}
        self.set_backend(backend, **backend_kws)

        solver_kws = solver_kws or {}
        self.set_solver(solver, **solver_kws)

    def __setattr__(self, key: str, value: tp.Any):
        if key == '_data':
            return super().__setattr__(key, value)
        try:
            self._data[key] = value
        except (AttributeError, ValueError): # _data not created yet or key not found in _data fields
            super().__setattr__(key, value)

    def __getattr__(self, key: str):
        if key == '_data':
            return super().__getattribute__(key)
        try:
            return self._data[key]
        except ValueError: # not found in fields
            return super().__getattribute__(key)

    @property
    def num(self) -> int:
        return self._data.shape[0]

    @abstractmethod
    def ode(self, **input_args) -> None:
        """
        The set of ODEs defining the dynamics of the model.

        .. note::

            ODE implements a scalar (element-wise) implementation
            of the model
        """

    def post(self, **post_args) -> None:
        """Post Processing

        Post-computation after each iteration of numerical update.

        For example, the hard reset for the IAF neuron must be implemented here.
        Another usage of this function could be the spike detection for
        conductance-based models.
        """
        raise NotImplementedError

    def recast(self) -> None:
        """Recast arrays to compatible formats"""
        self._data = cudaarray_to_cpu(self._data)
        self.states = self._data[[key for key in self.Default_States]]
        self.params = self._data[[key for key in self.Default_Params]]
        _gstates = self._data[[f"d_{key}" for key in self.Derivates]]
        _dtype = copy.deepcopy(_gstates.dtype)
        _dtype.names = self.Derivates
        self.gstates = self._data[[f"d_{key}" for key in self.Derivates]].view(_dtype)

    def reset(self) -> None:
        """Reset model states to initial and gstates to 0"""
        self.states[:] = self.initial_states
        self.gstates[:] = 0.0

    def clip(self, states: dict = None) -> None:
        """Clip state values by bounds"""
        states = self.states if states is None else states
        for var, bds in self.bounds.items():
            states[var].clip(*bds, out=states[var])

    def update(self, d_t: float, **input_args) -> None:
        """Update model value

        Arguments:
            d_t (float): time steps.
            input_args (dict): Arguments for input(s), must match
              call signature of :py:func:`Model.ode`.
        """
        self.solver.step(d_t, **input_args)
        try:
            self.post()
        except NotImplementedError:
            pass
        self.clip()
        for func in self.callbacks:
            func(self)

    def set_backend(self, new_backend: BackendMixin, **backend_options) -> None:
        assert new_backend == BackendMixin or issubclass(
            new_backend, BackendMixin
        ), err.NeuralBackendError(f"backend {new_backend} not understood")
        if not new_backend.is_backend_supported:
            raise err.NeuralBackendError(f"backend {new_backend} not supported.")

        new_supers = [new_backend, self.__class__] + [
            B
            for B in self.__class__.__bases__
            if not isinstance(B, BackendMixin) or B != object
        ]
        self.__class__ = type(self.__class__.__name__, tuple(new_supers), {})
        try:
            self.compile()  # compile model if the new backend has compile method defined
        except AttributeError:
            pass

        self.recast()  # recast array types

    def set_solver(self, new_solver: BaseSolver, **solver_options) -> None:
        if (
            new_solver == self.solver.__class__
            and self.solver.solver_options == solver_options
        ):
            return  # no-op

        self.solver = new_solver(self, **solver_options)

    def add_callback(self, callbacks: tp.Iterable[tp.Callable]) -> None:
        """Add callback to Model's `callbacks` list"""
        callbacks = list(callbacks)
        for func in callbacks:
            if not callable(func):
                raise err.NeuralModelError(
                    f"Function {func} is not callable but should be."
                )
            self.callbacks.append(func)

    def solve(
        self,
        t: np.ndarray,
        *,
        reset: bool = True,
        verbose: tp.Union[bool, str] = True,
        extra_callbacks: tp.Union[tp.Callable, tp.Iterable[tp.Callable]] = None,
        **input_args,
    ) -> tp.Dict:
        """Solve model equation for entire input

        Positional Arguments:
            t: 1d numpy array of time vector of the simulation

        Keyword-Only Arguments:
            reset: whether to reset the initial state value of the
              model to the values in :code:`Default_State`. Default to True.
            verbose: If is not `False` a progress bar will be created. If is `str`,
              the value will be set to the description of the progress bar.
            extra_callbacks: functions of the signature :code:`function(self)` that is
              executed at every step.

        Keyword Arguments:
            input_args: Key value pair of input arguments that matches the signature
              of the :func:`ode` function.

        Returns:
            An structured numpy array with shape :code:`(len(t), num)` and dtype
            :code:`self.dtypes['states']`. Contains simulation results.
        """
        # whether to reset initial state to `self.initial_states`
        if reset:
            self.reset()
        d_t = t[1] - t[0]

        # check external current dimension. It has to be either 1D array the
        # same shape as `t`, or a 2D array of shape `(len(t), self.num)`
        for var_name, stim in input_args.items():
            if var_name not in self.Inputs:
                raise err.NeuralModelError(
                    f"Extraneous input argument '{var_name}', "
                    f"support arguments are {self.Inputs.keys()}."
                )
            if np.isscalar(stim):
                continue
            stim = np.squeeze(stim)
            if not (
                (stim.ndim == 1 and len(stim) == len(t))
                or (stim.ndim == 2 and stim.shape == (len(t), self.num))
            ):
                raise err.NeuralModelError(
                    f"Stimulus '{var_name}' must be scalar or 1D/2D array of "
                    f"same length as t ({len(t)}) and num ({self.num}) "
                    f"in the second dimension, got {stim.shape} instead."
                )
            if stim.ndim == 1 and len(stim) == len(t):
                stim = np.repeat(stim[:, None], self.num, axis=1)
                # FIXME: This shouldn't be necessary.
                # the kernel should support broadcasting by itself.
                # maybe the compilation should be default.
            input_args[var_name] = stim

        # create stimuli generator
        stimuli = (
            {
                var: val if np.isscalar(val) else val[tt]
                for var, val in input_args.items()
                if var in self.Inputs
            }
            for tt in range(len(t) - 1)
        )
        # Register callback that is executed after every euler step.
        extra_callbacks = (
            [] if extra_callbacks is None else np.atleast_1d(extra_callbacks).tolist()
        )
        for f in extra_callbacks:
            if not callable(f):
                raise err.NeuralBackendError("Callback is not callable\n" f"{f}")

        # Solve
        res = np.zeros((len(t), self.num), dtype=self.states.dtype)
        res[0] = self.initial_states

        # run loop
        iters = enumerate(zip(t[:-1], stimuli), start=1)
        if verbose:
            iters = tqdm(
                iters,
                total=len(t),
                desc=verbose if isinstance(verbose, str) else "",
                dynamic_ncols=True,
            )

        for tt, (_t, _stim) in iters:
            self.update(d_t, **_stim)
            for _func in extra_callbacks:
                _func(self)
            res[tt] = self.states
        return res.T
