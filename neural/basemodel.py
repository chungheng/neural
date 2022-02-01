"""
Base model class for neurons and synapses.
"""
from tqdm.auto import tqdm
from abc import abstractmethod
from ast import Return
from warnings import warn
from inspect import getfullargspec
from dataclasses import dataclass
from functools import wraps
from numbers import Number
from types import ClassMethodDescriptorType
import typing as tp
import numpy as np
import numpy.typing as npt
import sympy as sp
from scipy.integrate import solve_ivp, OdeSolver, odeint
from scipy.interpolate import interp1d
from pycuda.gpuarray import GPUArray
from .backend import Backend
from . import types as tpe
from . import errors as err

class Model:
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
        clip: clip the state variables.
        forwardEuler: forward Euler method.

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
        solvers = dict()
        
        if cls.Default_Params is None:
            cls.Default_Params = dict()
        if cls.Default_States is None:
            cls.Default_States = dict()
        
        # structured dtype for states (states/gstates/bounds) and params
        for key, val in cls.Default_States.items():
            if hasattr(val, "__len__"):
                states[key] = val[0]
                if len(val) == 3:
                    bounds[key] = val[1:]
                else:
                    raise err.NeuralModelError(
                        f"Variable {key} should be a scalar of a iterable "
                        "of 3 or 4 elements (initial value, upper bound, lower bound[, dtype]) "
                        f"but {val} is given."
                    )
            else: # only values
                states[key] = val
            variables[key] = "states"
        cls.Default_Bounds = bounds
        cls.Default_States = states
        
        for key in cls.Default_Params:
            if key in cls.Default_States:
                raise err.NeuralModelError(
                    f"Parameters cannot have the same name as the States: '{key}'"
                )
        variables.update({key: "params" for key in cls.Default_Params})

        # TODO: Add Solvers

        # run ode once to get a list of state variables with derivative
        cls.Derivates = [key for key in states]
        obj = cls()
        for key in states:
            setattr(obj, key, 0.0)
            setattr(obj, "d_" + key, None)
        # call ode method, pass obj into argument as `self`
        try:
            obj.ode()
        except Exception as e:
            raise err.NeuralModelError(f"{cls}.ode() failed to execute") from e

        # record gradients
        d = {key: getattr(obj, "d_" + key) for key in states}
        # store gradients, filter out `None`
        cls.Derivates = [key for key, val in d.items() if val is not None]
        # store variables
        cls.Variables = variables

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
        self.gstates = {key: 0.0 for key in self.Derivates}
        self._check_dimensions()

        self.callbacks = []
        callback = [] if callback is None else callback
        self.add_callback(callback)

    def __setattr__(self, key: str, value: tp.Any):
        if key[:2] == "d_":
            assert key[2:] in self.gstates
            self.gstates[key[2:]] = value
            return

        if key in ["states", "params", "bounds"]:
            return super().__setattr__(key, value)

        if hasattr(self, 'states'):
            if key in self.states:
                self.states[key] = value
                return

        if hasattr(self, 'params'):
            if key in self.params:
                self.params[key] = value
                return

        super().__setattr__(key, value)

    def __getattr__(self, key: str):
        if key[:2] == "d_":
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

    def post(self) -> None:
        """Post Processing

        Post-computation after each iteration of numerical update.

        For example, the hard reset for the IAF neuron must be implemented here.
        Another usage of this function could be the spike detection for
        conductance-based models.
        """
        raise NotImplementedError

    def clip(self, states: dict = None) -> None:
        """Clip the State Variables

        Clip the state variables in-place after calling the numerical solver.

        The state variables are usually bounded, for example, binding
        variables are bounded between 0 and 1. However, numerical sovlers
        might cause the value of state variables exceed its bounds. A hard
        clip is forced here to ensure the state variables remain in the
        given bounds.
        """
        raise NotImplementedError

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
        self.step = None # TODO: Replace Step
        self.clip = None # TODO: Replace Clip
        self.solve = None # TODO: Replace Solve
        self.reset = None # TODO: Replace Reset if necessary

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

    def reset(self) -> None:
        """Reset Model
        
        Sets states to initial values, and sets gstates to 0.
        """
        for key, val in self.initial_states.items():
            if np.isscalar(val):
                self.states[key] = val
            else:
                self.states[key].fill(val)
        for key, val in self.gstates.items():
            if np.isscalar(val):
                self.gstates[key] = 0.
            else:
                self.gstates[key].fill(0.)

    @property
    def state_arr(self) -> np.ndarray:
        """State Vector for Batched ODE Solver

        This attribute stackes all state values into a
        :code:`(len(self.states), self.num)` shaped array. It is done for ODE
        solver to handle the state variable easily.
        """
        return np.vstack(list(self.states.values()))

    @state_arr.setter
    def state_arr(self, new_value) -> None:
        """Settting state_vector set states dictionary

        The setter and getter for state_arr is intended to ensure consistency
        between `self.states` and `self.state_arr`
        """
        for var_name, new_val in zip(self.states.keys(), new_value):
            self.states[var_name] = new_val

    def _check_dimensions(self) -> None:
        """Ensure consistent dimensions for all parameters and states"""

        for attr in ['params', 'states', 'gstates', 'initial_states']:
            # make sure vector-valued parameters have the same shape as the number
            # of components in the model
            dct = getattr(self, attr)
            for key, val in dct.items():
                if np.isscalar(val) or len(val) == 1:
                    if attr in ['states', 'gstates', 'initial_states']:
                        dct[key] = np.repeat(val, self.num)
                else:
                    assert len(val) == self.num, \
                        err.CompNeuroModelError(
                            f"{attr.capitalize()} '{key}'' should have scalar value or array of "
                            f"length num={self.num}, got {len(val)} instead."
                        )

    def solve(
        self,
        t: np.ndarray,
        *,
        solver: tpe.SupportedBackend = None,
        reset: bool = True,
        verbose: tp.Union[bool, str] = True,
        full_output: bool = False,
        callbacks: tp.Union[tp.Callable, tp.Iterable[tp.Callable]] = None,
        solver_kws: tp.Mapping[str, tp.Any] = None,
        **stimuli,
    ) -> tp.Union[tp.Dict, tp.Tuple[tp.Dict, tp.Any]]:
        """Solve model equation for entire input

        Positional Arguments:
            t: 1d numpy array of time vector of the simulation

        Keyword-Only Arguments:
            I_ext: external current driving the model

                .. deprecated:: 0.1.3
                    Use :code:`stimuli` keyword argument inputs instead.

            solver: Which ODE solver to use, defaults to the first entry in the
              :code:`Supported_Solvers` attribute.

                - `Euler`: Custom forward euler solver, default.
                - `odeint`: Use :py:mod:`scipy.integrate.odeint` which uses LSODA
                - `LSODA`: Use :py:mod:`scipy.integrate.odeint` which uses LSODA
                - `RK45/RK23/DOP853`: Use
                  :py:mod:`scipy.integrate.solve_ivp` with the specified method
                - :py:mod:`scipy.integrate.OdeSolver` instance: Use
                  :py:mod:`scipy.integrate.solve_ivp` with the provided custom solver

            reset_initial_state: whether to reset the initial state value of the
              model to the values in :code:`Default_State`. Default to True.
            verbose: If is not `False` a progress bar will be created. If is `str`,
              the value will be set to the description of the progress bar.
            full_output: whether to return the entire output from scipy's
              ode solvers.
            callbacks: functions of the signature :code:`function(self)` that is
              executed for :code:`solver=Euler` at every step.
            solver_kws: a dictionary containingarguments to be passed into the ode
              solvers if scipy solvers are used.

                .. seealso: :py:mod:`scipy.integrate.solve_ivp` and
                    :py:mod:`scipy.integrate.odeint`

        .. note::

            String names for :code:`solve_ivp` (RK45/RK23/DOP853)
            are case-sensitive but not for any other methods.
            Also note that the solvers can hang if the amplitude scale of
            :code:`I_ext` is too large.


        Keyword Arguments:
            stimuli: Key value pair of input arguments that matches the signature
              of the :func:`ode` function.

        Returns:
            Return dictionary of a 2-tuple depending on argument
            :code:`full_output`:

            - `False`: An dictionary of simulation results keyed by state
              variables and each entry is of shape :code:`(num, len(t))`
            - `True`: A 2-tuple where the first entry is as above, and the
              second entry is the ode result from either
              :py:mod:`scipy.integrate.odeint` or
              :py:mod:`scipy.integrate.solve_ivp`. The second entry will be
              :code:`None` if solver is :code:`Euler`
        """
        # validate solver
        if solver is None:
            solver = self.Supported_Solvers[0]
        if isinstance(solver, OdeSolver):
            pass
        else:
            if solver not in self.Supported_Solvers:
                raise err.NeuralModelError(
                    f"Solver '{solver}' not understood, must be one of "
                    f"{self.Supported_Solvers}."
                )
        solver_kws = {} if solver_kws is None else solver_kws

        # validate stimuli - check to make sure that the keyword arguments contain only
        # arguments that are relevant to the model input.
        if stimuli:
            _extraneous_input_args = set(stimuli.keys()) - set(self._input_args)
            _missing_input_args = set(self._input_args) - set(stimuli.keys())
            if _extraneous_input_args:
                raise err.CompNeuroModelError(
                    (
                        f"Extraneous input arguments '{_extraneous_input_args}' "
                        "treated as stimuli but are not found in the function "
                        f"definition of {self.__class__.__name__}.ode(), "
                        f"the only supported input variables are '{self._input_args}'"
                    )
                )
            if _missing_input_args:
                raise err.CompNeuroModelError(
                    f"Input argument '{_missing_input_args}' missing but are required "
                    f"by the {self.__class__.__name__}.ode() method. Please provide "
                    f"all inputs in '{self._input_args}'."
                )

        # whether to reset initial state to `self.initial_states`
        if reset:
            self.reset()

        # rescale time axis appropriately
        t_long = t * self.Time_Scale
        state_var_shape = self.state_arr.shape
        x0 = np.ravel(self.state_arr)
        d_t = t_long[1] - t_long[0]

        # check external current dimension. It has to be either 1D array the
        # same shape as `t`, or a 2D array of shape `(len(t), self.num)`
        if stimuli:
            for var_name, stim in stimuli.items():
                if stim.ndim == 1:
                    stim = np.repeat(stim[:, None], self.num, axis=-1)
                elif stim.ndim != 2:
                    raise err.CompNeuroModelError(
                        f"Stimulus '{var_name}' must be 1D or 2D"
                    )
                if len(stim) != len(t):
                    raise err.CompNeuroModelError(
                        f"Stimulus '{var_name}' first dimesion must be the same length as t"
                    )
                if stim.shape[1] > 1:
                    if stim.shape != (len(t_long), self.num):
                        raise err.CompNeuroModelError(
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
        if len(callbacks) > 0 and solver.lower() != "euler":
            warn(
                f"Callback only supported for Euler's method, got '{solver}'",
                err.CompNeuroWarning,
            )
            callbacks = None

        # Solve Euler's method
        if solver.lower() == "euler":  # solver euler
            res = np.zeros((len(t_long), len(self.state_arr), self.num))
            # run loop
            iters = enumerate(t_long)
            if verbose:
                iters = tqdm(
                    iters,
                    total=len(t_long),
                    desc=verbose if isinstance(verbose, str) else "",
                )

            for tt, _t in iters:
                _stim = {var_name: stim[tt] for var_name, stim in stimuli.items()}
                d_x = np.vstack(self.ode(_t, self.state_arr, **_stim))
                self.state_arr += d_t * d_x
                self.clip()
                res[tt] = self.state_arr
                if callbacks is not None:
                    for _func in callbacks:
                        _func(self)
            # move time axis to last so that we end up with shape
            # (len(self.states), self.num, len(t))
            res = np.moveaxis(res, 0, 2)
            res = {key: res[n] for n, key in enumerate(self.states.keys())}

            if full_output:
                return res, None
            return res

        # Solve IVP Methods
        if verbose:
            pbar = tqdm(
                total=len(t_long), desc=verbose if isinstance(verbose, str) else ""
            )

        # 1. create update function for IVP
        jacc_f = None
        if stimuli:  # has external input
            interpolators = {
                var_name: interp1d(
                    t_long, stim, axis=0, kind="zero", fill_value="extrapolate"
                )
                for var_name, stim in stimuli.items()
            }
            if self.jacobian is not None:
                # rewrite jacobian function to include invaluation at input value
                def jacc_f(t, states):  # pylint:disable=function-redefined
                    return self.jacobian(  # pylint:disable=not-callable
                        t,
                        states,
                        **{var: intp_f(t) for var, intp_f in interpolators.items()},
                    )

            # the update function interpolates the value of input at every
            # step `t`
            def update(t_eval, states):
                if verbose:
                    pbar.n = int((t_eval - t_long[0]) // d_t)
                    pbar.update()
                d_states = np.vstack(
                    self.ode(
                        t=t_eval,
                        states=np.reshape(states, state_var_shape),
                        **{
                            var: intp_f(t_eval) for var, intp_f in interpolators.items()
                        },
                    )
                )
                return d_states.ravel()

        else:  # no external input
            jacc_f = self.jacobian

            # if no current is provided, the system solves ode as defined
            def update(t_eval, states):
                if verbose:
                    pbar.n = int((t_eval - t_long[0]) // d_t)
                    pbar.update()
                d_states = np.vstack(
                    self.ode(states=np.reshape(states, state_var_shape), t=t_eval)
                )
                return d_states.ravel()

        # solver system
        ode_res_info = None
        res = np.zeros((len(t_long), len(self.state_arr), self.num))
        if isinstance(solver, OdeSolver):
            rtol = solver_kws.pop("rtol", 1e-8)
            ode_res = solve_ivp(
                update,
                t_span=(t_long.min(), t_long.max()),
                y0=x0,
                t_eval=t_long,
                method=solver,
                rtol=rtol,
                jac=jacc_f,
            )
            ode_res_info = ode_res
            res = ode_res.y.reshape((len(self.state_arr), -1, len(t_long)))
        elif solver.lower() in ["lsoda", "odeint"]:
            ode_res = odeint(
                update,
                y0=x0,
                t=t_long,
                tfirst=True,
                full_output=full_output,
                Dfun=jacc_f,
                **solver_kws,
            )
            if full_output:
                ode_res_y = ode_res[0]
                ode_res_info = ode_res[1]
                res = ode_res_y.T.reshape((len(self.state_arr), -1, len(t_long)))
            else:
                res = ode_res.T.reshape((len(self.state_arr), -1, len(t_long)))
        else:  # any IVP solver
            rtol = solver_kws.pop("rtol", 1e-8)
            options = {"rtol": rtol}
            if solver.lower() in IVP_SOLVER_WITH_JACC:
                options["jac"] = jacc_f
            ode_res = solve_ivp(
                update,
                t_span=(t_long.min(), t_long.max()),
                y0=x0,
                t_eval=t_long,
                method=solver,
                **options,
            )
            ode_res_info = ode_res
            res = ode_res.y.reshape((len(self.state_arr), -1, len(t_long)))

        res = {key: res[n] for n, key in enumerate(self.states.keys())}

        if verbose:
            pbar.update()
            pbar.close()

        if full_output:
            return res, ode_res_info
        return res