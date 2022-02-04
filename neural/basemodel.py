"""
Base model class for neurons and synapses.
"""
from abc import abstractmethod
from inspect import getfullargspec
from lib2to3.pytree import Base
import typing as tp
import numpy as np
from .solver import SOLVERS, BaseSolver, Euler
from tqdm.auto import tqdm
from . import types as tpe
from . import errors as err


class Model:
    """Base Model Class

    .. note::

        This class overrides :code:`__getattr__` and :code:`__setattr__`, and hence allows
        direct access to the subattributes contained in :code:`states` and :code:`params`,
        for example::

            # self.params = {'a': 1., 'b':1.}
            # self.states = {'s':0., 'x':0.}
            self.ds = self.a*(1-self.s) -self.b*self.s

    Class Attributes:
        Default_States (dict): The default value of the state variables.
          Each items represents the name (`key`) and the default value
          (`value`) of one state variables. If `value` is a tuple of three
          numbers, the first number is the default value, and the last two
          are the lower and the upper bound of the state variables.
        Default_Params (dict): The default value of the parameters. Each items
          represents the name (`key`) and the default value (`value`) of one
          parameters.
        Variables (dict): mapping of variable str name to type ['state', 'params']
        Inputs (dict): mapping of input name to default value as defined by 
          :py:func:`Model.ode`

    Attributes:
        states (np.ndarray): the state variables, updated by the ODE.
        params (np.ndarray): parameters of the model, can only be set during
          contrusction.
        gstates (np.ndarray): the gradient of the state variables.
        bounds (np.ndarray): lower and upper bounds of the state variables.
    """

    Default_States: tp.Dict = None
    Default_Params: tp.Dict = None
    Time_Scale: float = 1.0
    solver = Euler

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        bounds = dict()
        states = dict()
        variables = dict()

        if cls.Default_Params is None:
            cls.Default_Params = dict()
        if cls.Default_States is None:
            cls.Default_States = dict()

        dtypes = dict(
            states={var: np.float_ for var in cls.Default_States},
            gstates={var: np.float_ for var in cls.Default_States},
            params={var: np.float_ for var in cls.Default_Params},
            bounds={var: (np.float_, 2) for var in cls.Default_States},
        )
        # structured dtype for states (states/gstates/bounds) and params
        for key, val in cls.Default_States.items():
            states[key] = val if np.isscalar(val) else val[0]

            # handle custom dtypes
            # use numpy dtype if is numpy specific.
            # NOTE: if user wants to use integer type for a state, you must
            # specify it explicitly as Default_States=dict(var=np.int_(intial_value))
            if isinstance(states[key], np.generic):
                dtypes["states"][key] = val.dtype
                dtypes["gstates"][key] = val.dtype
            else:
                states[key] = dtypes["states"][key](states[key])

            if not np.isscalar(val):
                if len(val) != 3:
                    raise err.NeuralModelError(
                        f"Variable {key} should be a scalar of a iterable "
                        "of 3 elements (initial value, upper bound, lower bound) "
                        f"but {val} is given."
                    )
                bounds[key] = (
                    dtypes["states"][key](val[1]),
                    dtypes["states"][key](val[2]),
                )
            variables[key] = "states"
        cls.Default_Bounds = bounds

        # parse params
        for key, val in cls.Default_Params.items():
            if key in cls.Default_States:
                raise err.NeuralModelError(
                    f"Parameters cannot have the same name as the States: '{key}'"
                )
            if isinstance(val, np.generic):
                dtypes["params"][key] = val.dtype
            else:
                cls.Default_Params[key] = dtypes["params"][key](val)
        variables.update({key: "params" for key in cls.Default_Params})

        # store variables
        cls.Variables = variables

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
                inputs[key] = val

        cls.Inputs = inputs

        # create structured dtypes
        cls.dtypes = {
            attr: np.dtype(list(val.items()))
            for attr, val in dtypes.items()
        }
        cls.dtypes["bounds"] = np.dtype(
            [
                (var, cls.dtypes["bounds"][var])
                for var in bounds
            ]
        )
        # cast default states/params as numpy structured array
        cls.Default_States = np.asarray(
            tuple(states.values()), dtype=cls.dtypes["states"]
        )
        cls.Default_Bounds = np.asarray(
            tuple(bounds.values()), dtype=cls.dtypes["bounds"]
        )
        cls.Default_Params = np.asarray(
            tuple(cls.Default_Params.values()), dtype=cls.dtypes["params"]
        )

        # validate class ode definition
        cls.Derivates = [
            key for key in states
        ]  # temporarily set Derivates for all states
        obj = cls()
        for key in states:
            setattr(obj, key, 0.0)
            setattr(obj, "d_" + key, None)
        # call ode method, pass obj into argument as `self`
        try:
            obj.ode()
        except Exception as e:
            raise err.NeuralModelError(f"{cls}.ode failed to execute") from e

        # store state variables with gradients
        cls.Derivates = [
            var
            for var in states
            if hasattr(obj, f"d_{var}") 
            and np.isfinite(getattr(obj, f"d_{var}"))
        ]
        # cleanup gstates types to only include states with gradient defined
        cls.dtypes["gstates"] = np.dtype([
            (var, cls.dtypes["gstates"][var])
            for var in cls.Derivates
        ])

    def __init__(
        self,
        num: int = 1,
        callback: tp.Union[tp.Callable, tp.Iterable[tp.Callable]] = None,
        solver: BaseSolver = Euler,
        **kwargs,
    ):
        """
        Initialize the model.

        Arguments:
            num: number of components for the model
            callback: iterable of callback functions.
              callback functions should have model instance as first and
              only argument.

        Keyword Arguments:
            Are assumed to be values for states or parameters
        """
        self.num = num

        # set state variables and parameters
        self.params = self.Default_Params.copy()
        self.states = self.Default_States.copy()
        self.bounds = self.Default_Bounds.copy()
        # set additional variables
        for key, val in kwargs.items():
            if key in self.states.dtype.names:
                self.states[key] = val
            elif key in self.params.dtype.names:
                self.params[key] = val
            else:
                raise err.NeuralModelError(f"Unexpected state/variable '{key}'")

        self.initial_states = self.states.copy()
        self.gstates = np.zeros(self.num, dtype=self.dtypes["gstates"])
        self._check_dimensions()

        self.callbacks = []
        callback = [] if callback is None else callback
        self.add_callback(callback)
        self.set_solver(solver)

    def _check_dimensions(self) -> None:
        """Ensure consistent dimensions for all parameters and states"""

        for attr in ["params", "states", "gstates", "initial_states"]:
            # make sure vector-valued parameters have the same shape as the number
            # of components in the model
            arr = getattr(self, attr)
            if not (arr.size in [1, self.num] and arr.ndim in [0, 1]):
                raise err.NeuralModelError(
                    f"{attr.capitalize()} should be 0D/1D array of length 1 or num "
                    f"({self.num}), got {len(arr)} instead."
                )
            if attr == "params" or attr == "initial_states":
                continue
            if arr.size == 1:
                setattr(self, attr, np.repeat(arr, self.num))

    def __setattr__(self, key: str, value: tp.Any):
        if key.startswith("d_"):
            if key[2:] not in self.gstates.dtype.names:
                raise AttributeError(
                    f"Attribute {key} assumed to be gradient, but not found in Model.gstates"
                )
            self.gstates[key[2:]] = value
            return

        if key in ["states", "params", "bounds"]:
            return super().__setattr__(key, value)

        if hasattr(self, "states") and key in self.states.dtype.names:
            self.states[key] = value
            return

        if hasattr(self, "params") and key in self.params.dtype.names:
            self.params[key] = value
            return

        super().__setattr__(key, value)

    def __getattr__(self, key: str):
        if key.startswith("d_"):
            return self.gstates[key[2:]]

        if key in ["states", "params", "bounds"]:
            return super().__getattribute__(key)

        for attr in (self.states, self.params):
            if key in attr.dtype.names:
                return attr[key]

        return super().__getattribute__(key)

    @abstractmethod
    def ode(self, **input_args) -> None:
        """
        The set of ODEs defining the dynamics of the model.
        """

    def post(self, **post_args) -> None:
        """Post Processing

        Post-computation after each iteration of numerical update.

        For example, the hard reset for the IAF neuron must be implemented here.
        Another usage of this function could be the spike detection for
        conductance-based models.
        """

    def update(self, d_t: float, **input_args) -> None:
        """Update model value

        Arguments:
            d_t (float): time steps.
            input_args (dict): Arguments for input(s), must match
              call signature of :py:func:`Model.ode`.
        """
        self.solver.step(d_t, **input_args)
        self.post()
        self.clip()
        for func in self.callbacks:
            func(self)

    def reset(self) -> None:
        """Reset state values.
        
        Sets states to initial values, and sets gstates to 0.
        """
        if callable(reset:= getattr(self.solver, "reset", None)):
            reset()
            return
        self.states.fill(self.initial_states)
        self.gstates.fill(0.)

    def clip(self, states: dict = None) -> None:
        """Clip the State Variables

        Clip the state variables in-place after calling the numerical solver.

        The state variables are usually bounded, for example, binding
        variables are bounded between 0 and 1. However, numerical sovlers
        might cause the value of state variables exceed its bounds. A hard
        clip is forced here to ensure the state variables remain in the
        given bounds.
        """
        states = self.states if states is None else states
        if callable(clip:= getattr(self.solver, "clip", None)):
            clip(states=states)
            return
        for var in self.bounds.dtype.names:
            states[var].clip(*self.bounds[var], out=states[var])

    def set_solver(self, new_solver: BaseSolver, **solver_options) -> None:
        self.solver = new_solver(self, **solver_options)
        new_solver.recast_arrays(self)

    def add_callback(self, callbacks: tp.Iterable[tp.Callable]) -> None:
        """Add callback to Model's `callbacks` list"""
        if not hasattr(callbacks, "__len__"):
            callbacks = [
                callbacks,
            ]
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
                raise err.NeuralModelError(f"Extraneous input argument '{var_name}'")
            if np.isscalar(stim):
                continue
            stim = np.squeeze(stim)
            if not (
                (stim.ndim == 1 and len(stim) == len(t)) 
                or 
                (stim.ndim == 2 and stim.shape == (len(t), self.num))
            ):
                raise err.NeuralModelError(
                    f"Stimulus '{var_name}' must be scalar or 1D/2D array of "
                    f"same length as t ({len(t)}) and num ({self.num}) "
                    f"in the second dimension, got {stim.shape} instead."
                )
            input_args[var_name] = stim

        # create stimuli generator
        stimuli = (
            {
                var: val if np.isscalar(val) else val[tt]
                for var, val in input_args.items()
                if var in self.Inputs
            }
            for tt in range(len(t)-1)
        )
        # Register callback that is executed after every euler step.
        extra_callbacks = (
            [] if extra_callbacks is None else np.atleast_1d(extra_callbacks).tolist()
        )
        for f in extra_callbacks:
            if not callable(f):
                raise err.NeuralBackendError("Callback is not callable\n" f"{f}")

        # Solve
        res = np.zeros((len(t), self.num), dtype=self.dtypes["states"])
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
            res[tt][:] = self.states
        return res.T
