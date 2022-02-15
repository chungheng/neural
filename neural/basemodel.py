# pylint:disable=method-hidden
"""
Base model class for neurons and synapses.
"""
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
    create_empty,
    cudaarray_to_cpu,
    get_array_module,
    isarray,
    iscudaarray,
    cuda_fill,
)
from .backend import Backend, NumbaCPUBackend, NumbaCUDABackend
from ._method_dispatcher import MethodDispatcher


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
    solver: BaseSolver = Euler
    backend: Backend = None
    Supported_Backends: tp.Iterable[Backend] = (
        Backend,
        NumbaCPUBackend,
        NumbaCUDABackend,
    )

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

        vis = FindDerivates()
        vis.visit(ast.parse(textwrap.dedent(inspect.getsource(cls.ode))))
        cls.Derivates = [var for var in vis.Derivates if var in cls.Default_States]

        # all the methods that are decorated as DispatchByBackend in Model
        # should also be decorated in the subclasses. If not decorated already,
        # we manually decorate it.
        for method in ["ode", "post", "clip", "reset", "recast"]:
            # check if method is decorated
            func = getattr(cls, method)
            if not hasattr(func, "register"):
                func = MethodDispatcher(func)
                MethodDispatcher.__set_name__(func, cls, method)
                setattr(cls, method, func)

    def __init__(
        self,
        num: int = 1,
        callback: tp.Union[tp.Callable, tp.Iterable[tp.Callable]] = None,
        backend: Backend = Backend,
        backend_kws: dict = None,
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

        # set state variables and parameters
        self.num = 1  # num will be set in the first set_num call
        self.params = {
            key: np.atleast_1d(val) for key, val in self.Default_Params.items()
        }
        self.states = {
            key: np.atleast_1d(val) for key, val in self.Default_States.items()
        }
        self.bounds = copy.deepcopy(self.Default_Bounds)
        self.gstates = {
            var: np.zeros_like(arr)
            for var, arr in self.states.items()
            if var in self.Derivates
        }
        self.initial_states = copy.deepcopy(self.states)
        self.set_num(num)

        # set additional variables
        for key, val in kwargs.items():
            if key not in self.states and key not in self.params:
                raise err.NeuralModelError(f"Unexpected state or param '{key}'")
            for name in ["states", "params"]:
                dct = getattr(self, name)
                if key in dct:
                    ref_dtype = dct[key].dtype
                    if hasattr(val, "dtype") and val.dtype != ref_dtype:
                        warn(
                            (
                                f"Input for {name} {key} has dtype {val.dtype} that is "
                                f"different from that default dtype {ref_dtype}."
                                "Casting array to the default dtype."
                            ),
                            err.NeuralModelWarning,
                        )
                    dct[key][:] = val

        # update initial states
        self.initial_states = copy.deepcopy(self.states)
        self.callbacks = []
        callback = [] if callback is None else callback
        self.add_callback(callback)

        backend_kws = backend_kws or {}
        self.set_backend(backend, **backend_kws)
        solver_kws = solver_kws or {}
        self.set_solver(solver, **solver_kws)

    def set_num(self, num: int = None, keep_idx: tp.Iterable[int] = None) -> None:
        """Set number of model

        Arguments:
            num: number of model components
            keep_idx: slice original array down using these indices
              if num is smaller than current num
        """
        if not (isinstance(num, int) and num >= 1):
            raise err.NeuralModelError("num must be an integer greater than 0.")
        if num < self.num and keep_idx is None:
            raise err.NeuralModelError(
                "when new num is smaller than current num, keep_idx must be specified "
                "to perform the required slicing"
            )
        if num > self.num and not (self.num == 1 or isinstance(keep_idx, int)):
            raise err.NeuralModelError(
                "when new num is larger than current num, current num must either be "
                "1 or keep_idx must be specified to repeat the kept slice to new num"
            )
        for attr in ["params", "states", "gstates", "initial_states"]:
            for key, arr in (dct := getattr(self, attr)).items():
                if num < self.num:
                    dct[key] = arr[keep_idx]
                elif num > self.num:
                    if keep_idx is not None:
                        arr = arr[keep_idx]
                    if (module := get_array_module(arr)) is None:  # scalar
                        arr = np.full((num,), arr)
                    try:
                        arr = module.repeat(arr, num)  # numpy, cupy
                    except AttributeError:  # pycuda, no repeat method
                        if iscudaarray(arr):
                            val = cudaarray_to_cpu(arr).item()
                        elif isarray(arr):
                            val = arr.item()
                        else:
                            val = arr
                        new_arr = create_empty((num,), like=arr)
                        cuda_fill(new_arr, val)
                        arr = new_arr
                    except Exception as e:
                        raise err.NeuralModelError(
                            f"Cannot create array for {attr}.{key}"
                        ) from e
                dct[key] = arr
        self.num = num

    def __setattr__(self, key: str, value: tp.Any):
        if key.startswith("d_"):
            try:
                self.gstates[key[2:]] = value
            except KeyError as e:
                raise AttributeError(
                    f"Attribute {key} assumed to be gradient, but not found in Model.gstates"
                ) from e
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
        for attr in ["states", "gstates", "bounds", "params"]:
            for key, arr in (dct := getattr(self, attr)).items():
                dct[key] = cudaarray_to_cpu(arr)

    def reset(self) -> None:
        for attr in self.states:
            self.states[attr][:] = self.initial_states[attr]
        for attr in self.gstates:
            self.gstates[attr][:] = 0.0

    def clip(self, states: dict = None) -> None:
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

    def set_backend(self, new_backend: Backend, **backend_options) -> None:
        assert new_backend == Backend or issubclass(
            new_backend, Backend
        ), err.NeuralBackendError(f"backend {new_backend} not understood")

        if self.backend.__class__ == new_backend:
            return
        self.backend = new_backend(self, **backend_options)
        try:
            self.backend.compile()
        except AttributeError:
            pass

        for method in ["ode", "post", "clip", "reset", "recast"]:
            if callable(func := getattr(self.backend, method, None)):
                getattr(self, method).register(new_backend, func)
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
