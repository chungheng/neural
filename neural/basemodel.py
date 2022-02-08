"""
Base model class for neurons and synapses.
"""
import copy
from abc import abstractmethod
from warnings import warn
from inspect import getfullargspec
import typing as tp
import numpy as np
from .solver import BaseSolver, Euler
from tqdm.auto import tqdm
from . import types as tpe
from . import errors as err
from .utils.array import isarray
from .codegen.parsedmodel import ParsedModel


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
        cls.Time_Scale = np.float_(cls.Time_Scale)

        bounds = dict()
        states = dict()
        variables = dict()

        if cls.Default_Params is None:
            cls.Default_Params = dict()
        if cls.Default_States is None:
            cls.Default_States = dict()

        # structured dtype for states (states/gstates/bounds) and params
        for key, val in cls.Default_States.items():
            dtype = val.dtype if isinstance(val, np.generic) else np.float_
            states[key] = dtype(val) if np.isscalar(val) else dtype(val[0])

            if not np.isscalar(val):
                if len(val) != 3:
                    raise err.NeuralModelError(
                        f"Variable {key} should be a scalar of a iterable "
                        "of 3 elements (initial value, upper bound, lower bound) "
                        f"but {val} is given."
                    )
                bounds[key] = np.asarray(val[1:], dtype=dtype)
            variables[key] = "states"
        cls.Default_States = states
        cls.Default_Bounds = bounds

        # parse params
        for key, val in cls.Default_Params.items():
            dtype = val.dtype if isinstance(val, np.generic) else np.float_
            cls.Default_Params[key] = dtype(val)
            if key in cls.Default_States:
                raise err.NeuralModelError(
                    f"Parameters cannot have the same name as the States: '{key}'"
                )
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
                dtype = val.dtype if isinstance(val, np.generic) else np.float_
                inputs[key] = dtype(val)

        cls.Inputs = inputs

        # validate class ode definition
        # temporarily set Derivates for all states
        cls.Derivates = [key for key in states]
        obj = cls()
        # set all gstates to None to filter out which gstates are actually set
        obj.gstates = {var: None for var in obj.gstates}
        # call ode method, pass obj into argument as `self`
        try:
            obj.ode()
        except Exception as e:
            raise err.NeuralModelError(f"{cls.__name__}.ode failed to execute") from e
        try:
            obj.post()
        except Exception as e:
            raise err.NeuralModelError(f"{cls.__name__}.post failed to execute") from e

        # store state variables with gradients
        cls.Derivates = [var for var, val in obj.gstates.items() if val is not None]

    def __init__(
        self,
        num: int = 1,
        callback: tp.Union[tp.Callable, tp.Iterable[tp.Callable]] = None,
        solver: BaseSolver = Euler,
        solver_kws: dict = None,
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
        self.params = copy.deepcopy(self.Default_Params)
        self.states = copy.deepcopy(self.Default_States)
        self.bounds = copy.deepcopy(self.Default_Bounds)
        # set additional variables
        for key, val in kwargs.items():
            if key not in self.states and key not in self.params:
                raise err.NeuralModelError(f"Unexpected state/variable '{key}'")
            for name in ['states', 'params']:
                dct = getattr(self, name)
                if key in dct:
                    ref_dtype = dct[key].dtype
                    if isarray(val) and len(val) != num:
                        raise err.NeuralModelError(
                            f"Value for state {key} has length {len(val)}, expect {num}"
                        )
                    if hasattr(val, 'dtype') and val.dtype != ref_dtype:
                        warn(
                            (
                                f"Input for {name} {key} has dtype {val.dtype} that is "
                                f"different from that default dtype {ref_dtype}."
                                "Casting array to the default dtype."
                            ),
                            err.NeuralModelWarning
                        )
                        val = val.astype(ref_dtype)
                    if np.isscalar(val):
                        val = ref_dtype.type(val)
                    dct[key] = val
                    continue

        self.initial_states = copy.deepcopy(self.states)
        self.gstates = {
            var: np.zeros_like(arr)
            for var, arr in self.states.items()
            if var in self.Derivates
        }
        self._check_dimensions()

        self.callbacks = []
        callback = [] if callback is None else callback
        self.add_callback(callback)

        # create symbolic model
        try:
            self.symbolic = ParsedModel(self)
        except Exception as e:
            self.symbolic = None

        self._jacobian = None
        self.get_jacobian()

        solver_kws = solver_kws or {}
        self.set_solver(solver, **solver_kws)

    def _check_dimensions(self) -> None:
        """Ensure consistent dimensions for all parameters and states"""

        for attr in ["params", "states", "gstates", "initial_states"]:
            # make sure vector-valued parameters have the same shape as the number
            # of components in the model
            for key, arr in (dct := getattr(self, attr)).items():
                if np.isscalar(arr) and attr in ["params", "initial_states"]:
                    continue
                arr = np.asarray(arr)
                if not (arr.size in [1, self.num] and arr.ndim in [0, 1]):
                    raise err.NeuralModelError(
                        f"{attr.capitalize()}['{key}'] should be 0D/1D array of length 1 or num "
                        f"({self.num}), got {len(arr)} instead: {arr}"
                    )
                if attr in ["params", "initial_states"]:
                    continue
                if arr.size == 1:
                    dct[key] = np.repeat(arr, self.num)

    def __setattr__(self, key: str, value: tp.Any):
        if key.startswith("d_"):
            if key[2:] not in self.gstates:
                raise AttributeError(
                    f"Attribute {key} assumed to be gradient, but not found in Model.gstates"
                )
            self.gstates[key[2:]] = value
            return

        if key in ["states", "params", "bounds"]:
            return super().__setattr__(key, value)

        if hasattr(self, "states") and key in self.states:
            self.states[key] = value
            return

        if hasattr(self, "params") and key in self.params:
            self.params[key] = value
            return

        super().__setattr__(key, value)

    def __getattr__(self, key: str):
        if key.startswith("d_"):
            return self.gstates[key[2:]]

        if key in ["states", "params", "bounds"]:
            return super().__getattribute__(key)

        for attr in (self.states, self.params):
            if key in attr:
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
        if callable(reset := getattr(self.solver, "reset", None)):
            reset()
            return
        for var in self.states:
            self.states[var].fill(self.initial_states[var])
        for var in self.gstates:
            self.gstates[var].fill(0.0)

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
        if callable(clip := getattr(self.solver, "clip", None)):
            clip(states=states)
            return
        for var, bds in self.bounds.items():
            states[var].clip(*bds, out=states[var])

    def set_solver(self, new_solver: BaseSolver, **solver_options) -> None:
        if (
            new_solver == self.solver.__class__
            and self.solver.solver_options == solver_options
        ):
            return  # no-op
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
        res = {var: np.zeros((len(t), self.num)) for var in self.states}
        for var, val in self.initial_states.items():
            res[var][0] = val

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
            for var, val in self.states.items():
                res[var][tt][:] = val
        return {var: val.T for var, val in res.items()}

    def get_jacobian(self) -> tp.Callable:
        """Compute Jacobian of Model

        .. note::

            Differing from :py:func:`Model.jacobian`, this function will always
            `re-compute` jacobian, including re-parsing the model. This is
            provided in case the model parameter has been changed in-place
            and the jacobian needs to be updated.

        Returns:
            A callable :code:`jacc_f(t, states, **input_args)` that returns a 2D numpy
            array corresponding to the jacobian of the model.
        """
        if self.symbolic is None:
            return None
        # set the jacobian for model
        self._jacobian = self.symbolic.get_lambidified_jacobian()
        return self._jacobian

    @property
    def jacobian(self) -> tp.Callable:
        """Compute or return cached jacobian of the model

        .. note::

            You can override jacobian definition in child classes to enforce
            a jacobian

        .. seealso:: :py:func:`neural.Model.get_jacobian`

        Returns:
            A callable :code:`jacc_f(t, states, I_ext)` that returns a 2D numpy
            array corresponding to the jacobian of the model. For model that does
            not require `I_ext` input, the callable's call signature is
            :code:`jacc_f(t, states, I_ext)`.
        """
        if self._jacobian is not None:
            return self._jacobian
        return self.get_jacobian()
