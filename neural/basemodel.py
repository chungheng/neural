"""
Base model class for neurons and synapses.
"""
from abc import abstractmethod
import typing as tp
import numpy as np
from scipy.integrate import solve_ivp
from pycuda.gpuarray import GPUArray
from .backend import Backend
from .errors import NeuralModelError, NeuralModelWarning
from inspect import getfullargspec

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


class ModelMetaClass(type):
    """Model MetaClass

    Model MetaClass provies a custom :code:`__new__` method that:

    1. parses the class attributes of the Model Class
    2. runs state updates to record state variables and gradients
    3. set :code:`Time_Scale` to default 1 if not provided
    4. set solver
    5. check to makesure that only keyword arguments are allowed in :code:`ode` and
       :code:`post` methods.
    """

    def __new__(cls, clsname, bases, dct):
        bounds = dict()
        states = dict()
        variables = dict()
        if "Default_States" in dct:
            for key, val in dct["Default_States"].items():
                if hasattr(val, "__len__"):
                    if len(val) != 3:
                        raise NeuralModelError(
                            f"Variable {key} should be a scalar of a iterable "
                            "of 3 elements (initial value, upper bound, lower bound) "
                            f"but {val} is given."
                        )
                    bounds[key] = val[1:]
                    states[key] = val[0]
                else:
                    states[key] = val
                variables[key] = "states"
        dct["Default_Bounds"] = bounds
        dct["Default_States"] = states

        if "Default_Params" not in dct:
            dct["Default_Params"] = dict()
        variables.update({key: "params" for key in dct["Default_Params"]})

        # run ode once to get a list of state variables with derivative
        obj = type(clsname, (object,), dct["Default_Params"])()
        for key in states:
            setattr(obj, key, 0.0)
            setattr(obj, "d_" + key, None)
        # call ode method, pass obj into argument as `self`
        dct["ode"](obj)
        # record gradients
        d = {key: getattr(obj, "d_" + key) for key in states}
        # store gradients, filter out `None`
        dct["Derivates"] = [key for key, val in d.items() if val is not None]
        # store variables
        dct["Variables"] = variables

        if "Time_Scale" not in dct:
            dct["Time_Scale"] = 1.0

        if clsname == "Model":
            solvers = dict()
            for key, val in dct.items():
                if callable(val) and hasattr(val, "_solver_names"):
                    for name in val._solver_names:
                        solvers[name] = val.__name__
            dct["solver_alias"] = solvers

        inputs = dict()
        func_list = [x for x in ["ode", "post"] if x in dct]
        for key in func_list:
            argspec = getfullargspec(dct[key])
            if argspec.defaults is None:
                continue
            if argspec.varargs is not None:
                raise NeuralModelError(
                    f"Variable positional argument is not allowed in {clsname}.{key}"
                )
            if getattr(argspec, "varkw", None) is not None:
                raise NeuralModelError(
                    f"Variable keyword argument is not allowed in {clsname}.{key}"
                )
            for val, key in zip(argspec.defaults[::-1], argspec.args[::-1]):
                inputs[key] = val
        dct["Inputs"] = inputs

        return super(ModelMetaClass, cls).__new__(cls, clsname, bases, dct)


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


class Model(metaclass=ModelMetaClass):
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
        Default_Inters (dict): Optional. The default value of the intermediate
            variables. Each items represents the name (`key`) and the default value of
            one intermediate variable. (`value`) of one state variables.
        Default_Bounds (dict): The lower and the upper bound of the state
            variables. It is created through the `ModelMetaClass`.
        Variables (dict):
        Inputs (dict):

    Attributes:
        states (dict): the state variables, updated by the ODE.
        params (dict): parameters of the model, can only be set during
            contrusction.
        gstates (dict): the gradient of the state variables.
        bounds (dict): lower and upper bounds of the state variables.
    """

    def __init__(
        self,
        solver: str = "forward_euler",
        callback: tp.Union[tp.Callable, tp.Iterable[tp.Callable]] = None,
        optimize: bool = False,
        **kwargs,
    ):
        """
        Initialize the model.

        Keyword arguments:
            optimize: optimize the `ode` function.
            callback: callback(s) for model
            solver: which solver to use

        Additional Keyword Arguments are assumed to be values for states or parameters
        """
        optimize = optimize and (Backend is not None)
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
                raise NeuralModelError(f"Unexpected state/variable '{key}'")

        self.initial_states = self.states.copy()
        self.gstates = {key: 0.0 for key in self.Derivates}

        # set numerical solver
        if "scipy_ivp" in solver:  # scipy_ivp can take form `scipy_ivp:method`
            all_scipy_solvers = ("RK45", "RK23", "Radau", "BDF", "LSODA")
            _scipy_solver = solver.split("scipy_ivp:")[-1]
            self._scipy_solver = (
                _scipy_solver if _scipy_solver in all_scipy_solvers else "RK45"
            )
            solver = self.solver_alias["scipy_ivp"]
        else:
            if solver in self.solver_alias:
                solver = self.solver_alias[solver]
            else:
                raise NeuralModelError(f"Unexpected Solver '{solver}'")
        self.solver = getattr(self, solver)

        self._update = self._scalar_update
        self._reset = self._scalar_reset
        self.callbacks = []
        self.add_callback(callback)

        # optimize the ode function
        if optimize:
            self.compile(backend="scalar")

    def compile(self, **kwargs) -> None:
        """
        compile the cuda kernel.

        Keyword Arguments:
            num (int): The number of units for CUDA kernel excution.
            dtype (type): The default type of floating point for CUDA.
        """
        self.backend = Backend(model=self, **kwargs)

        for attr in ("ode", "post"):
            if hasattr(self.backend, attr):
                setattr(self, attr, getattr(self.backend, attr))

        for attr in ("data", "reset", "update"):
            if hasattr(self.backend, attr):
                setattr(self, "_" + attr, getattr(self.backend, attr))

    def add_callback(self, callbacks) -> None:
        """Add callback to Model's `callbacks` list"""
        if not hasattr(callbacks, "__len__"):
            callbacks = [
                callbacks,
            ]
        for func in callbacks:
            if not callable(func):
                raise NeuralModelError(
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

    def _ode_wrapper(self, states: dict = None, gstates: dict = None, **kwargs):
        """
        A wrapper for calling `ode` with arbitrary variable than `self.states`

        Arguments:
            states: state variables.
            gstates: gradient of state variables.
        """
        if states is not None:
            _states = self.states
            self.states = states
        if gstates is not None:
            _gstates = self.gstates
            self.gstates = gstates

        self.ode(**kwargs)

        if states is not None:
            self.states = _states

        if gstates is not None:
            self.gstates = _gstates
        else:
            return self.gstates

    def post(self):
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
                raise NeuralModelError(
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
            raise NeuralModelError("'to_graph' requires 'pycodegen'") from e
        except Exception as e:
            raise NeuralModelError("Unknown Error to 'Model.to_graph' call") from e
        return VariableAnalyzer(cls).to_graph(local=local)

    @classmethod
    def to_latex(cls):
        """Convert Circuit Equation to Latex

        Generate latex source code for the  model
        """
        try:
            from .codegen.symbolic import SympyGenerator
        except ImportError as e:
            raise NeuralModelError("'to_latex' requires 'pycodegen'") from e
        except Exception as e:
            raise NeuralModelError("Unknown Error to 'Model.to_latex' call") from e
        return SympyGenerator(cls).latex_src

    def _increment(
        self, d_t: float, states: dict, out_states: dict = None, **kwargs
    ) -> dict:
        """
        Compute the increment of state variables.

        This function is used for advanced numerical methods.

        Arguments:
            d_t (float): time steps.
            states (dict): state variables.
        """
        if out_states is None:
            out_states = states.copy()

        gstates = self._ode_wrapper(states, **kwargs)

        for key in gstates:
            out_states[key] = d_t * gstates[key]

        return out_states

    def _forward_euler(
        self, d_t: float, states: dict, out_states: dict = None, **kwargs
    ) -> dict:
        """
        Forward Euler method with arbitrary variable than `self.states`.

        This function is used for advanced numerical methods.

        Arguments:
            d_t (float): time steps.
            states (dict): state variables.
        """
        if out_states is None:
            out_states = states.copy()

        self._increment(d_t, states, out_states, **kwargs)

        for key in out_states:
            out_states[key] += states[key]

        self.clip(out_states)

        return out_states

    @register_solver("euler", "forward")
    def forward_euler(self, d_t: float, **kwargs) -> None:
        """
        Forward Euler method.

        Arguments:
            d_t (float): time steps.
        """
        self.ode(**kwargs)

        for key in self.gstates:
            self.states[key] += d_t * self.gstates[key]
        self.clip()

    @register_solver("mid")
    def midpoint(self, d_t: float, **kwargs) -> None:
        """
        Implicit Midpoint method.

        Arguments:
            d_t (float): time steps.
        """
        _states = self.states.copy()

        self._forward_euler(0.5 * d_t, self.states, _states, **kwargs)
        self._forward_euler(d_t, _states, self.states, **kwargs)

    @register_solver("heun")
    def heun(self, d_t: float, **kwargs) -> None:
        """
        Heun's method.

        Arguments:
            d_t (float): time steps.
        """
        incr1 = self._increment(d_t, self.states, **kwargs)
        tmp = _dict_add_(self.states, incr1)
        incr2 = self._increment(d_t, tmp, **kwargs)

        for key in self.states.keys():
            self.states[key] += 0.5 * incr1[key] + 0.5 * incr2[key]
        self.clip()

    @register_solver("rk4")
    def runge_kutta_4(self, d_t: float, **kwargs) -> None:
        """
        Runge Kutta method.

        Arguments:
            d_t (float): time steps.
        """
        tmp = self.states.copy()

        k1 = self._increment(d_t, self.states, **kwargs)

        _dict_add_scalar_(self.states, k1, 0.5, out=tmp)
        self.clip(tmp)
        k2 = self._increment(d_t, tmp, **kwargs)

        _dict_add_scalar_(self.states, k2, 0.5, out=tmp)
        self.clip(tmp)
        k3 = self._increment(d_t, tmp, **kwargs)

        _dict_add_(self.states, k3, out=tmp)
        self.clip(tmp)
        k4 = self._increment(d_t, tmp, **kwargs)

        for key in self.states.keys():
            incr = (k1[key] + 2.0 * k2[key] + 2.0 * k3[key] + k4[key]) / 6.0
            self.states[key] += incr
        self.clip()

    @register_solver("scipy_ivp")
    def scipy_ivp(self, d_t: float, t: np.ndarray = None, **kwargs) -> None:
        """
        Wrapper for scipy.integrate.solve_ivp

        Arguments:
            d_t (float): time steps.
        """
        solver = kwargs.pop("solver", self._scipy_solver)
        # ensure that the dictionary keys are matched
        keys = list(self.gstates.keys())
        vectorized_states = [self.states[k] for k in keys]

        def f(states, t):  # note that the arguments are dummies,
            # not used anywhere in the body of the function
            res = self._ode_wrapper(**kwargs)
            return [res[k] for k in keys]

        res = solve_ivp(
            f, [0, d_t], vectorized_states, method=solver, t_eval=[d_t], events=None
        )
        if res is not None:
            self.states.update(
                {k: res.y[:, -1][n] for n, k in enumerate(keys)}
            )  # indexing res.y[:, -1] incase multiple steps are returned
        self.clip()
        # TODO: `events` can be used for spike detection

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
                raise NeuralModelError(
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

    def __setattr__(self, key: str, value: tp.Any):
        if key[:2] == "d_":
            assert key[2:] in self.gstates
            self.gstates[key[2:]] = value
            return

        if key in ["states", "params", "bounds"]:
            return super(Model, self).__setattr__(key, value)

        for attr in (self.states, self.params):
            if key in attr:
                attr[key] = value
                return

        super(Model, self).__setattr__(key, value)

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

        return super(Model, self).__getattribute__(key)
