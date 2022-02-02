"""
Base model class for neurons and synapses.
"""
from abc import abstractmethod
from inspect import getfullargspec
import typing as tp
import numpy as np
from .backend import Backend as BackendMixIn
from tqdm.auto import tqdm
from . import types as tpe
from . import errors as err

class Model(BackendMixIn):
    """Base Model Class

    This class overrides :code:`__getattr__` and :code:`__setattr__`, and hence allows
    direct access to the subattributes contained in :code:`states` and :code:`params`,
    for example::

        # self.params = {'a': 1., 'b':1.}
        # self.states = {'s':0., 'x':0.}
        self.ds = self.a*(1-self.s) -self.b*self.s

    Methods:
        ode: a set of ODEs defining the dynamics of the system.
        update: wrapper function to the numeric solver for each time step.
        post: computation after calling the numerical solver.

    Class Attributes:
        Default_States (dict): The default value of the state variables.
            Each items represents the name (`key`) and the default value
            (`value`) of one state variables. If `value` is a tuple of three
            numbers, the first number is the default value, and the last two
            are the lower and the upper bound of the state variables.
        Default_Params (dict): The default value of the parameters. Each items
            represents the name (`key`) and the default value (`value`) of one
            parameters.
        Variables (dict):
        Inputs (dict):

    Attributes:
        states (dict): the state variables, updated by the ODE.
        params (dict): parameters of the model, can only be set during
            contrusction.
        gstates (dict): the gradient of the state variables.
        bounds (dict): lower and upper bounds of the state variables.
    """
    Default_States: tp.Dict = None
    Default_Params: tp.Dict = None
    Solvers: tp.Iterable = ('Euler', 'RK23', 'RK45') # TODO: Standardize this
    Backends: tp.Iterable = ['numpy', 'cupy', 'pycuda'] # TODO: Standardize this
    _backend: tpe.SupportedBackend = "numpy"
    Time_Scale: float = 1.

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
        )
        # structured dtype for states (states/gstates/bounds) and params
        for key, val in cls.Default_States.items():
            states[key] = val if np.isscalar(val) else val[0]

            # handle custom dtypes
            # use numpy dtype if is numpy specific.
            # NOTE: if user wants to use integer type for a state, you must
            # specify it explicitly as Default_States=dict(var=np.int_(intial_value))
            if isinstance(states[key], np.generic):
                dtypes['states'][key] = val.dtype
                dtypes['gstates'][key] = val.dtype
            else:
                states[key] = dtypes['states'][key]( states[key] )

            if not np.isscalar(val):
                assert len(val) == 3, \
                    err.NeuralModelError(
                        f"Variable {key} should be a scalar of a iterable "
                        "of 3 elements (initial value, upper bound, lower bound) "
                        f"but {val} is given."
                    )
                bounds[key] = (
                    dtypes['states'][key](val[1]),
                    dtypes['states'][key](val[2])
                )
            variables[key] = "states"
        cls.Default_Bounds = bounds

        # parse params
        for key, val in cls.Default_Params.items():
            assert key not in cls.Default_States, \
                err.NeuralModelError(
                    f"Parameters cannot have the same name as the States: '{key}'"
                )
            if isinstance(val, np.generic):
                dtypes['params'][key] = val.dtype
            else:
                cls.Default_Params[key] = dtypes['params'][key]( val )
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
        cls.dtypes = {attr:np.dtype(list(val.items())) for attr, val in dtypes.items()}

        # cast default states/params as numpy structured array
        cls.Default_States = np.asarray([tuple(states.values())], dtype=cls.dtypes['states'])
        cls.Default_Params = np.asarray([tuple(cls.Default_Params.values())], dtype=cls.dtypes['params'])

        # validate class ode definition
        cls.Derivates = [key for key in states] # temporarily set Derivates for all states
        obj = cls()
        for key in states:
            setattr(obj, key, 0.0)
            setattr(obj, "d_" + key, None)
        # call ode method, pass obj into argument as `self`
        try:
            obj.ode()
        except Exception as e:
            raise err.NeuralModelError(f"{cls}.ode() failed to execute") from e

        # store state variables with gradients
        cls.Derivates = [
            var
            for var in states
            if hasattr(obj, f"d_{var}")
            and np.isfinite(getattr(obj, f"d_{var}"))
        ]


    def __init__(
        self,
        num: int = 1,
        callback: tp.Union[tp.Callable, tp.Iterable[tp.Callable]] = None,
        **kwargs,
    ):
        """
        Initialize the model.

        Keyword arguments:
            callback: callback(s) for model

        Additional Keyword Arguments are assumed to be values for states or parameters
        """
        self.num = num

        # set state variables and parameters
        self.params = self.Default_Params.copy()
        self.states = self.Default_States.copy()
        self.bounds = self.Default_Bounds.copy()
        # set additional variables
        for key, val in kwargs.items():
            if key in self.states:
                self.states[key] = val
            elif key in self.params:
                self.params[key] = val
            else:
                raise err.NeuralModelError(f"Unexpected state/variable '{key}'")

        self.initial_states = self.states.copy()
        self.gstates = np.zeros(self.num, dtype=self.dtypes['gstates'])
        self._check_dimensions()

        self.callbacks = []
        callback = [] if callback is None else callback
        self.add_callback(callback)

    def __setattr__(self, key: str, value: tp.Any):
        if key[:2] == "d_":
            assert key[2:] in self.gstates.dtype.names
            self.gstates[key[2:]] = value
            return

        if key in ["states", "params", "bounds"]:
            return super().__setattr__(key, value)

        if hasattr(self, 'states'):
            if key in self.states.dtype.names:
                self.states[key] = value
                return

        if hasattr(self, 'params'):
            if key in self.params.dtype.names:
                self.params[key] = value
                return

        super().__setattr__(key, value)

    def __getattr__(self, key: str):
        if key[:2] == "d_":
            return self.gstates[key[2:]]

        if key in ["states", "params", "bounds"]:
            return super().__getattribute__(key)

        for attr in (self.states, self.params):
            if key in attr.dtype.names:
                return attr[key]

        return super().__getattribute__(key)

    def update(self, d_t: float, **input_args) -> None:
        """Update model value

        Arguments:
            d_t (float): time steps.
            input_args (dict): Arguments for input(s), must match
              call signature of :py:func:`Model.ode`.
        """
        self.step(d_t, **input_args)
        try:
            self.post()
        except NotImplementedError:
            pass
        try:
            self.clip()
        except NotImplementedError:
            pass
        for func in self.callbacks:
            func()

    def solve(self,
        t: np.ndarray,
        *,
        solver: tpe.SupportedBackend = None,
        reset: bool = True,
        verbose: tp.Union[bool, str] = True,
        callbacks: tp.Union[tp.Callable, tp.Iterable[tp.Callable]] = None,
        solver_kws: tp.Mapping[str, tp.Any] = None,
        **stimuli,
    ) -> tp.Dict:
        """Solve model equation for entire input

        Positional Arguments:
            t: 1d numpy array of time vector of the simulation

        Keyword-Only Arguments:
            solver: Which ODE solver to use, defaults to the first entry in the
              :code:`Supported_Solvers` attribute.

                - `Euler`: Custom forward euler solver, default.
                - `odeint`: Use :py:mod:`scipy.integrate.odeint` which uses LSODA
                - `LSODA`: Use :py:mod:`scipy.integrate.odeint` which uses LSODA
                - `RK45/RK23/DOP853`: Use
                  :py:mod:`scipy.integrate.solve_ivp` with the specified method
                - :py:mod:`scipy.integrate.OdeSolver` instance: Use
                  :py:mod:`scipy.integrate.solve_ivp` with the provided custom solver

            reset: whether to reset the initial state value of the
              model to the values in :code:`Default_State`. Default to True.
            verbose: If is not `False` a progress bar will be created. If is `str`,
              the value will be set to the description of the progress bar.
            callbacks: functions of the signature :code:`function(self)` that is
              executed for :code:`solver=Euler` at every step.
            solver_kws: a dictionary containingarguments to be passed into the ode
              solvers if the solver accepts arguments.

        Keyword Arguments:
            stimuli: Key value pair of input arguments that matches the signature
              of the :func:`ode` function.

        Returns:
            An dictionary of simulation results keyed by state
            variables and each entry is of shape :code:`(num, len(t))`
        """
        # validate solver
        if solver is None:
            solver = self.Solvers[0]
        assert solver in self.Solvers, \
            err.NeuralModelError(
                f"Solver '{solver}' not understood, must be one of "
                f"{self.Solvers}."
            )
        solver_kws = {} if solver_kws is None else solver_kws

        # Validate Stimuli
        # check to make sure that the keyword arguments contain only
        # arguments that are relevant to the model input.
        if stimuli:
            if _extraneous_input_args := set(stimuli.keys()) - set(self.Inputs):
                raise err.NeuralModelError(
                    (
                        f"Extraneous input arguments '{_extraneous_input_args}' "
                        "treated as stimuli but are not found in the function "
                        f"definition of {self.__class__.__name__}.ode(), "
                        f"the only supported input variables are '{self.Inputs}'"
                    )
                )
            if _missing_input_args := set(self.Inputs) - set(stimuli.keys()):
                raise err.NeuralModelError(
                    f"Input argument '{_missing_input_args}' missing but are required "
                    f"by the {self.__class__.__name__}.ode() method. Please provide "
                    f"all inputs in '{self.Inputs}'."
                )

        # whether to reset initial state to `self.initial_states`
        if reset:
            self.reset()

        # rescale time axis appropriately
        t_long = t * self.Time_Scale
        d_t = t_long[1] - t_long[0]

        # check external current dimension. It has to be either 1D array the
        # same shape as `t`, or a 2D array of shape `(len(t), self.num)`
        if stimuli:
            for var_name, stim in stimuli.items():
                if stim.ndim == 1:
                    stim = np.repeat(stim[:, None], self.num, axis=-1)
                elif stim.ndim != 2:
                    raise err.NeuralModelError(
                        f"Stimulus '{var_name}' must be 1D or 2D array"
                    )
                if len(stim) != len(t):
                    raise err.NeuralModelError(
                        f"Stimulus '{var_name}' first dimesion must be the same length as t"
                    )
                if stim.shape[1] > 1 and stim.shape != (len(t_long), self.num):
                    raise err.NeuralModelError(
                        f"Stimulus '{var_name}' expects shape ({len(t_long)},{self.num}), "
                        f"got {stim.shape}"
                    )
                stimuli[var_name] = stim

        # Register callback that is executed after every euler step.
        callbacks = [] if callbacks is None else np.atleast_1d(callbacks).tolist()
        for f in callbacks:
            if not callable(f):
                raise err.CompNeuroModelError("Callback is not callable\n" f"{f}")
        callbacks = tuple(list(self.callbacks) + callbacks)

        # Solve
        res = np.zeros((len(t_long), self.num), dtype=self.dtypes['states'])
        # run loop
        iters = enumerate(t_long)
        if verbose:
            iters = tqdm(
                iters,
                total=len(t_long),
                desc=verbose if isinstance(verbose, str) else "",
                dynamic_ncols=True
            )

        for tt, _t in iters:
            _stim = {var_name: stim[tt] for var_name, stim in stimuli.items()}
            self.update(d_t, **_stim)
            if callbacks is not None:
                for _func in callbacks:
                    _func(self)
            res[tt][:] = self.states
        return res

    @abstractmethod
    def ode(self, **input_args) -> None:
        """
        The set of ODEs defining the dynamics of the model.
        """

    def post(self) -> None:
        """Post Processing

        Post-computation after each iteration of numerical update.

        For example, the hard reset for the IAF neuron must be implemented here.
        Another usage of this function could be the spike detection for
        conductance-based models.
        """
        raise NotImplementedError

    def _check_dimensions(self) -> None:
        """Ensure consistent dimensions for all parameters and states"""

        for attr in ['params', 'states', 'gstates', 'initial_states']:
            # make sure vector-valued parameters have the same shape as the number
            # of components in the model
            arr = getattr(self, attr)
            assert len(arr) in [1, self.num], \
                err.NeuralModelError(
                    f"{attr.capitalize()} should be length 1 or num ({self.num}), "
                    f"got {len(arr)} instead."
                )
            if attr == 'params':
                continue
            if len(arr) == 1:
                setattr(self, attr, np.repeat(arr, self.num))

    @property
    def backend(self) -> tpe.SupportedBackend:
        return self._backend

    @backend.setter
    def backend(self, new_backend: tpe.SupportedBackend):
        if new_backend == self.backend:
            return
        if new_backend not in self.Backends:
            raise err.NeuralModelError(
                f"Backend '{new_backend}' not supported. Must be one of '{self.Backends}'"
            )

        # instantiate backend class with "self" as input,
        # route methods from backend to methods of "self"
        if hasattr(new_backend, "update"):
            self.update = new_backend.update
        else:
            self.step = new_backend.step # TODO: Replace Step
            self.clip = new_backend.clip # TODO: Replace Clip
            self.solve = new_backend.solve # TODO: Replace Solve
            self.reset = new_backend.reset # TODO: Replace Reset if necessary

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

