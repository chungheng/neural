"""
Base model class for neurons and synapses.
"""
from abc import abstractmethod
from dataclasses import dataclass

from functools import wraps
from inspect import getfullargspec
from numbers import Number
from types import ClassMethodDescriptorType
import typing as tp
import numpy as np
import numpy.typing as npt
import sympy as sp
from scipy.integrate import solve_ivp
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
    Solvers: tp.Iterable = ('Euler', 'RK23', 'RK45')
    backend: tpe.SupportedBackend = "numpy"
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
        dtypes = dict(states=[], params=[])
        default_dtype = np.float_
        for key, val in cls.Default_States.items():
            dtype = default_dtype # default dtype of state
            if hasattr(val, "__len__"):
                states[key] = val[0]
                if len(val) == 3:
                    bounds[key] = val[1:]
                elif len(val) == 4:
                    bounds[key] = val[1:-1]
                    dtype = val[-1]
                else:
                    raise err.NeuralModelError(
                        f"Variable {key} should be a scalar of a iterable "
                        "of 3 or 4 elements (initial value, upper bound, lower bound[, dtype]) "
                        f"but {val} is given."
                    )
            else: # only values
                states[key] = val
            variables[key] = "states"
            dtypes['states'].append((key, dtype))
        cls.Default_Bounds = bounds
        cls.Default_States = states
        
        for key in cls.Default_Params:
            if key in cls.Default_States:
                raise err.NeuralModelError(
                    f"Parameters cannot have the same name as the States: '{key}'"
                )
        variables.update({key: "params" for key in cls.Default_Params})

        # # register solvers
        # if cls.solver_alias is None:
        #     cls.solver_alias = dict()
        # for key, val in cls.__dict__.items():
        #     if callable(val) and hasattr(val, "_solver_names"):
        #         for name in val._solver_names:
        #             cls.solver_alias[name] = val.__name__

        cls.Derivates = [key for key in states]
        # run ode once to get a list of state variables with derivative
        obj = cls(**cls.Default_Params)
        for key in states:
            setattr(obj, key, 0.0)
            setattr(obj, "d_" + key, None)
        # call ode method, pass obj into argument as `self`
        cls.ode(obj)
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
        callback: tp.Union[tp.Callable, tp.Iterable[tp.Callable]] = None,
        **kwargs,
    ):
        """
        Initialize the model.

        Keyword arguments:
            callback: callback(s) for model

        Additional Keyword Arguments are assumed to be values for states or parameters
        """
        if callback is None:
            callback = []

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
        
        # # set numerical solver
        # if "scipy_ivp" in solver:  # scipy_ivp can take form `scipy_ivp:method`
        #     all_scipy_solvers = ("RK45", "RK23", "Radau", "BDF", "LSODA")
        #     _scipy_solver = solver.split("scipy_ivp:")[-1]
        #     self._scipy_solver = (
        #         _scipy_solver if _scipy_solver in all_scipy_solvers else "RK45"
        #     )
        #     solver = self.solver_alias["scipy_ivp"]
        # else:
        #     if solver in self.solver_alias:
        #         solver = self.solver_alias[solver]
        #     else:
        #         raise err.NeuralModelError(f"Unexpected Solver '{solver}'")
        # self.solver = getattr(self, solver)

        self._update = self._scalar_update
        self._reset = self._scalar_reset
        self.callbacks = []
        self.add_callback(callback)


    def compile(self, backend: tpe.SupportedBackend = None, **kwargs) -> None:
        """
        compile the ODE kernel and create aliases of backend methods for model

        Keyword Arguments:
            backend: backend to use
            num: The number of units for CUDA kernel excution.
            dtype: The default type of floating point for CUDA.
        """
        self.backend = Backend(model=self, backend=backend, **kwargs)

        # create alias of backend methods in model
        for attr in ("ode", "post"):
            if hasattr(self.backend, attr):
                setattr(self, attr, getattr(self.backend, attr))

        for attr in ("data", "reset", "update"):
            if hasattr(self.backend, attr):
                setattr(self, "_" + attr, getattr(self.backend, attr))

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

    def reset(self, **kwargs) -> None:
        """
        reset state and intermediate variables to their initial condition.

        Notes:
            This function is a wrapper to the underlying `_reset`.
        """
        # '_reset' depends on the 'backend'
        self._reset(**kwargs)

    def update(self, d_t: float, **kwargs) -> None:
        """
        Wrapper function for each iteration of update.

        ``update`` is a proxy to one of ``_cpu_update`` or ``_cuda_update``.

        Arguments:
            d_t: time steps.
            kwargs: Arguments for input(s) or other purposes. For
            example, one can use an extra boolean flag to indicate the
            period for counting spikes.

        Notes:
            The signature of the function does not specify _stimulus_
            arguments. However, the developer should provide the stimulus
            to the model, ex. `input` or `spike`. If mulitple stimuli are
            required, the developer could specify them as `input1` and `input2`.

            This function is a wrapper to the underlying `_update`.
        """
        # '_update' depends on the 'backend'
        self._update(d_t * self.Time_Scale, **kwargs)
        for func in self.callbacks:
            func()

    @abstractmethod
    def ode(self, **kwargs) -> None:
        """
        The set of ODEs defining the dynamics of the model.

        TODO: enable using different state variables than self.states
        """

    def post(self) -> None:
        """Post Processing

        Post-computation after each iteration of numerical update.

        For example, the hard reset for the IAF neuron must be implemented here.
        Another usage of this function could be the spike detection for
        conductance-based models.
        """

    def clip(self, states: dict = None) -> None:
        """Clip the State Variables

        Clip the state variables in-place after calling the numerical solver.

        The state variables are usually bounded, for example, binding
        variables are bounded between 0 and 1. However, numerical sovlers
        might cause the value of state variables exceed its bounds. A hard
        clip is forced here to ensure the state variables remain in the
        given bounds.
        """
        if states is None:
            states = self.states

        for key, val in self.bounds.items():
            try:
                self.backend.clip(states[key], val[0], val[1])
            except AttributeError:
                states.update({key: np.clip(states[key], val[0], val[1])})
            except Exception as e:
                raise err.NeuralModelError(
                    f"Model {self} clip state '{key}' unknown error"
                ) from e

    @classmethod
    def to_graph(cls, local: bool = False):
        """Convert Circuit to Graph

        Generate block diagram of the model

        Parameters:
            local: Whether to include local variables or not.
        """
        try:
            from .codegen.symbolic import VariableAnalyzer
        except ImportError as e:
            raise err.NeuralModelError("'to_graph' requires 'pycodegen'") from e
        except Exception as e:
            raise err.NeuralModelError("Unknown Error to 'Model.to_graph' call") from e
        return VariableAnalyzer(cls).to_graph(local=local)

    @classmethod
    def to_latex(cls):
        """Convert Circuit Equation to Latex

        Generate latex source code for the  model
        """
        try:
            from .codegen.symbolic import SympyGenerator
        except ImportError as e:
            raise err.NeuralModelError("'to_latex' requires 'pycodegen'") from e
        except Exception as e:
            raise err.NeuralModelError("Unknown Error to 'Model.to_latex' call") from e
        return SympyGenerator(cls).latex_src

    def _scalar_update(self, d_t: float, **kwargs) -> None:
        """
        Wrapper function for running solver on CPU.

        Arguments:
            d_t (float): time steps.
            kwargs (dict): Arguments for input(s) or other purposes. For
            example, one can use an extra boolean flag to indicate the
            period for counting spikes.

        Notes:
            The signature of the function does not specify _stimulus_
            arguments. However, the developer should provide the stimulus
            to the model, ex. `input` or `spike`. If mulitple stimuli are
            required, the developer could specify them as `input1` and `input2`.
        """
        for key, val in kwargs.items():
            if isinstance(val, GPUArray):
                kwargs[key] = val.get()
        self.solver(d_t, **kwargs)
        self.post()

    def _scalar_reset(self, **kwargs) -> None:
        for key, val in kwargs.items():
            if key not in self.Variables:
                raise err.NeuralModelError(
                    f"Attempting to reset key={key} but not in self.Va"
                )
            attr = self.Variables[key]
            if attr == "states":
                key = "initial_" + key
            dct = getattr(self, attr)
            dct[key] = val
        for key, val in self.initial_states.items():
            self.states[key] = val
        for key in self.gstates.keys():
            self.gstates[key] = 0.0
    
    @classmethod
    def register_solver(cls, *args):
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

    def __setattr__(self, key: str, value: tp.Any):
        if key[:2] == "d_":
            assert key[2:] in self.gstates
            self.gstates[key[2:]] = value
            return

        if key in ["states", "params", "bounds"]:
            return super().__setattr__(key, value)

        for attr in (self.states, self.params):
            if key in attr:
                attr[key] = value
                return

        super().__setattr__(key, value)

    def __getattr__(self, key: str):
        if "_data" in self.__dict__ and key in self._data:
            return self._data[key]
        if key[:2] == "d_":
            return self.gstates[key[2:]]

        if key in ["states", "params", "bounds"]:
            return getattr(self, key)

        for attr in (self.states, self.params):
            if key in attr:
                return attr[key]

        return super().__getattribute__(key)