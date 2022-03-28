"""
Network module for constructing an abitrary circuit of neurons and synapses.

Examples:

.. code-block:: python

    >>> nn = Network()
    >>> iaf = nn.add(IAF, bias=0., c=10., record=['v'])
    >>> syn = nn.add(AlphaSynapse, ar=1e-1, stimulus=iaf.spike)
    >>> hhn = nn.add(HodgkinHuxley, stimulus=syn.I)
    >>> nn.input(s=iaf.stimulus)
    >>> nn.compile(dtype=dtype)
    >>> nn.run(dt, s=numpy.random.rand(10000))
"""
import inspect
from functools import reduce
from numbers import Number
from warnings import warn
import typing as tp
import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

# pylint:disable=relative-beyond-top-level
from ..basemodel import Model
from ..recorder import Recorder
from ..backend import NumbaCPUBackendMixin
from .. import errors as err
from .. import types as tpe
from .. import utils
from ..solver import SOLVERS, BaseSolver, Euler
from ..utils import isarray, isiterator

# pylint:enable=relative-beyond-top-level


class Symbol(object):
    def __init__(self, container: tpe.Container, key: str):
        self.container = container
        self.key = key

    def __getitem__(self, given: str) -> tp.Any:
        attr = getattr(self.container.recorder, self.key)
        return attr.__getitem__(given)


class Input:
    """Input Object for Neural Network

    Note:
        An Input object :code:`inp` can be updated using either of the 2 methods:

        1. :code:`value = next(inp)`
        2. :code:`inp.step(); value = inp.value`

        The latter method is useful if an input object's value is to be read by
        multiple containers.

    FIXME: If num = 1, the output is broadcasted
    """

    def __init__(self, num: int = 1, name: str = None):
        self.num = num
        self.name = name
        self.data = None
        self.steps = 0
        self.iter = None
        self.latex_src = "External stimulus"
        self.graph_src = utils.model.MINIMUM_PNG
        self.value = None

    def __call__(self, data: tp.Union[npt.ArrayLike, tp.Iterable]):
        if not isarray(data) and not isiterator(data):
            raise err.NeuralNetworkInputError(
                f"Input '{self.name}' data must be either array or iterator, "
                f"got {type(data)} instead."
            )

        steps = len(data) if isarray(data) else 0
        if isarray(data):
            if not data.flags.c_contiguous:
                raise err.NeuralNetworkInputError(
                    f"Input '{self.name}' ndarray must be c-contiguous"
                )
            if data.ndim not in [1, 2]:
                raise err.NeuralNetworkInputError(
                    f"Input '{self.name}' is given data of shape={data.shape}, only up-to "
                    "2D data is supported currently."
                )
            if data.ndim == 2 and self.num != data.shape[1]:
                raise err.NeuralNetworkInputError(
                    f"Input '{self.name}' is specified with num={self.num} but was "
                    f"given data of shape={data.shape}"
                )
            steps = len(data)

        self.data = data
        self.steps = steps
        self.reset()
        return self

    def step(self) -> tpe.ScalarOrArray:
        self.value = next(self)

    def __next__(self) -> tpe.ScalarOrArray:
        return next(self.iter)

    def reset(self) -> None:
        if hasattr(self.data, "__iter__"):
            self.iter = iter(self.data)
        else:
            self.iter = (x for x in self.data)


class Container(object):
    """
    A wrapper holds an Model instance with symbolic reference to its variables.

    Examples:

    .. code-block:: python

        >>> hhn = Container(HodgkinHuxley())
        >>> hhn.v  # reference to hhn.states['v']
    """

    def __init__(
        self,
        obj: tp.Union[tpe.Model, tpe.Symbol, Number, tpe.Input],
        name: str = "",
    ):
        self.obj = obj
        self.name = name
        self.vars = dict()
        self.inputs = dict()
        self.recorder = None
        self._rec = []

    @property
    def num(self) -> int:
        try:
            return self.obj.num
        except AttributeError:
            return None

    def __repr__(self) -> str:
        return f"Container[{self.obj}] - num {self.num}"

    def __call__(self, **kwargs) -> tpe.Container:
        """Setup input connection to Container

        Calling a container has the following behaviors:
          1. If container wraps a `Model` module, then:
            1. If called with a key that is part of the container variable,
              the value of that variable is set to the corresponding value
            2. If called with a key that is part of the input,
              check if the value is an acceptable type. Set the input if
              acceptable, else raise an error.
            3. If neither, then an error is raised.
          2. If container does not wrap a `Model` instance, then assume that
            the value is an input
        """
        for key, val in kwargs.items():
            if isinstance(self.obj, Model):
                if key in self.obj.Variables:
                    setattr(self.obj, key, val)
                elif key in self.obj.Inputs:
                    if not isinstance(val, (Symbol, Number, Input)):
                        raise err.NeuralContainerError(
                            f"Container {self.name} is called with value {val} that "
                            "is of a type not understood. Should be Symbol, Number of "
                            "Input"
                        )
                    self.inputs[key] = val
                else:
                    raise err.NeuralContainerError(
                        f"Attempting to set '{key}' of container but the "
                        "it is neither a variable nor an input."
                    )
            else:
                if not isinstance(val, (Symbol, Number, Input)):
                    raise err.NeuralContainerError(
                        f"Container {self.name} is called with value {val} that is "
                        "of a type not understood. Should be Symbol, Number of Input"
                    )
                self.inputs[key] = val
        return self

    def __getattr__(self, key: str) -> Symbol:
        if key in self.vars:
            return self.vars[key]
        try:
            _ = getattr(self.obj, key)
            self.vars[key] = Symbol(self, key)
            return self.vars[key]
        except:
            return super(Container, self).__getattribute__(key)

    @property
    def latex_src(self):
        return self._get_latex()

    @property
    def graph_src(self):
        return self._get_graph()

    def record(self, *args: tp.Iterable[str]) -> None:
        """Cache Arguments to be Recorded

        Notes:
            This method only tests if the specified arguments are found in the object
            to be recorded from. To actaully create the `Recorder` instance, call the
            `set_recorder` method after the recorded variables are cached in the
            `self._rec` list.
        """
        for arg in args:
            try:
                _ = getattr(self.obj, arg)
            except AttributeError:
                msg = err.NeuralRecorderWarning(
                    f"Attribute {arg} not found in {self.obj}, skipping"
                )
                warn(msg)
                continue
            if arg not in self._rec:
                self._rec.append(arg)

    def set_recorder(
        self, steps: int, rate: int = 1, gpu_bufsize: int = 500
    ) -> Recorder:
        """Create Recorder Instance

        Arguments:
            steps: total number of steps to record
            rate: sample rate at which the results are recorded
            gpu_bufsize: number of steps that is buffered on the gpu side.

        Returns:
            :code:`Recorder` instance or :code:`None` if no variable
            is set to be recorded.
        """
        if not self._rec:
            self.recorder = None
        elif (
            (self.recorder is None)
            or (self.recorder.total_steps != steps)
            or (set(self.recorder.dct.keys()) != set(self._rec))
        ):
            self.recorder = Recorder(
                self.obj, self._rec, steps, gpu_bufsize=gpu_bufsize, rate=rate
            )
        return self.recorder

    def _get_latex(self) -> str:
        latex_src = f"{self.obj.__class__.__name__}:<br><br>"
        # FIXME: use updated API

        # if isinstance(self.obj, Model):
        #     sg = SympyGenerator(self.obj)
        #     latex_src += sg.latex_src
        #     variables = [rf"\({x}\)" for x in sg.signature]
        #     latex_src += "<br>Input: " + ", ".join(variables)
        #     variables = []
        #     for _k, _v in sg.variables.items():
        #         if (_v.type in ["state", "intermediate"]) and _v.integral == None:
        #             variables.append(rf"\({_k}\)")
        #     latex_src += "<br>Variables: " + ", ".join(variables)

        return latex_src

    def _get_graph(self) -> bytes:
        if isinstance(self.obj, Model):
            return utils.model.to_graph(self.obj)
        return utils.model.MINIMUM_PNG

    @classmethod
    def isacceptable(cls, module_or_obj) -> bool:
        """Check if a custom module or object is acceptable as Container

        Modules are accepted if they have an update method.
        """
        return hasattr(module_or_obj, "update") and callable(module_or_obj.update)


class Network:
    """Neural Network Object"""

    def __init__(
        self, solver: tpe.Solver = Euler, backend: tpe.Backend = NumbaCPUBackendMixin
    ):
        self.containers = dict()
        self.inputs = dict()
        self.solver = self.validate_solver(solver)
        self.backend = backend
        self._iscompiled = False

    @classmethod
    def validate_solver(cls, solver: tp.Union[str, BaseSolver]) -> BaseSolver:
        """Validate solver"""
        if isinstance(solver, str):
            if (solver := getattr(SOLVERS, solver, None)) is None:
                raise err.NeuralNetworkError(
                    f"Solver not found in supported solvers: '{solver}'"
                )
        else:
            if not issubclass(solver, BaseSolver):
                raise err.NeuralNetworkError(
                    "Solver must be a subclass of neural.solver.BaseSolver"
                )
        return solver

    def input(self, num: int = 1, name: str = None) -> Input:
        """Create input object"""
        name = name or f"input{len(self.inputs)}"
        self.inputs[name] = Input(num=num, name=name)
        self._iscompiled = False
        return self.inputs[name]

    def add(
        self,
        module: tpe.Model,
        num: int = None,
        name: str = None,
        record: tp.Iterable[str] = None,
        solver: tpe.Solver = None,
        backend: tpe.Backend = None,
        **module_args,
    ) -> Container:
        solver_kws = module_args.pop("solver_kws", {})
        solver = self.validate_solver(solver or self.solver)
        backend = backend or self.backend

        if (name := name or f"obj{len(self.containers)}") in self.containers:
            raise err.NeuralNetworkError(f"Duplicate container name: '{name}'")

        if isinstance(module, Model):
            module.set_solver(solver, **solver_kws)
            module.set_backend(backend)
            obj = module
            if num is not None and obj.num != num:
                raise err.NeuralContainerError(
                    f"num argument ({num}) does not equal to num of model ({obj.num})"
                )
        elif issubclass(module, Model):
            obj = module(
                solver=solver,
                num=num,
                solver_kws=solver_kws,
                backend=backend,
                **module_args,
            )
        elif inspect.isclass(module):
            if not Container.isacceptable(module):
                raise err.NeuralNetworkError(
                    f"{module} is not an acceptable module type for Container"
                )
            module_args["size"] = num
            obj = module(**module_args)
        else:
            raise err.NeuralNetworkError(
                f"{module} is not accepted as module for Container. "
                "Must be instance of, subclass of or supports update() "
                "API of neural.basemodel.Model."
            )

        self.containers[name] = container = Container(obj, name=name)
        if record is not None:
            container.record(*list(record))

        self._iscompiled = False
        return container

    def run(
        self,
        d_t: float,
        steps: int = 0,
        rate: int = 1,
        verbose: str = None,
        solver: tp.Union[str, BaseSolver] = None,
        gpu_bufsize: int = 500,
    ) -> None:
        """Run Network

        Keyword Arguments:
            d_t: step size in second
            steps: number of steps to run, inferred from input if not specified
            rate: frequency of recording output, default to recording every step
            verbose: Content to show for progressbar, set to `None`(default) to disable.
            solver: solver to use
            gpu_bufsize: gpu buffer size for recorder before transfering to cpu. only applicable
              for solvers that use gpu
        """
        if not self._iscompiled:
            raise err.NeuralNetworkCompileError(
                "Please compile before running the network."
            )

        if solver is not None:
            solver = self.validate_solver(solver)
            for name, cont in self.containers.items():
                cont.obj.set_solver(solver)

        # calculate number of steps
        steps = reduce(max, [input.steps for input in self.inputs.values()], steps)
        for name, val in self.inputs.items():
            if val.steps == 0:
                warn(f"Input '{name}' has 0 steps", err.NeuralNetworkWarning)

        # create recorders
        for c in self.containers.values():
            _ = c.set_recorder(steps, rate, gpu_bufsize=gpu_bufsize)

        # reset everything
        self.reset()

        # create iterator for simulation loop
        iterator = range(steps)
        if verbose is not None:
            if isinstance(verbose, str):
                iterator = tqdm(iterator, total=steps, desc=verbose, dynamic_ncols=True)
            else:
                iterator = tqdm(iterator, total=steps, dynamic_ncols=True)

        for _ in iterator:
            self.update(d_t)

    def reset(self) -> None:
        """Reset everything"""
        # reset recorders
        for c in self.containers.values():
            if c.recorder is not None:
                c.recorder.reset()

        # reset Model variables
        for c in self.containers.values():
            if hasattr(c.obj, "reset") and callable(c.obj.reset):
                c.obj.reset()

        # reset inputs
        for inp in self.inputs.values():
            inp.reset()

    def update(self, d_t: float) -> None:
        """Update Network and all components

        Arguments:
            dt: time step
            idx: time index of the update for recorder
        """
        # update inputs
        for c in self.inputs.values():
            c.step()

        # update containers
        for c in self.containers.values():
            input_args = {}
            for key, val in c.inputs.items():
                if isinstance(val, Symbol):
                    input_args[key] = getattr(val.container.obj, val.key)
                elif isinstance(val, Input):
                    input_args[key] = val.value
                elif isinstance(val, Number):
                    input_args[key] = val
                else:
                    raise err.NeuralNetworkUpdateError(
                        f"Input '{key}' of container '{c}' has invalid type: {val}"
                    )

            try:
                if isinstance(c.obj, Model):
                    c.obj.update(d_t, **input_args)
                else:
                    c.obj.update(**input_args)
            except Exception as e:
                raise err.NeuralNetworkUpdateError(
                    f"Update Failed for Model [{c.obj}]"
                ) from e
        # update recorders
        for c in self.containers.values():
            if c.recorder is not None:
                c.recorder.update()

    def compile(self, dtype: npt.DTypeLike = np.float_, debug: bool = False) -> None:
        """Compile the module
        compile backend for every model

        FIXME: Is this function still needed?
        """
        # for c in self.containers.values():
        #     dct = {}
        #     for key, val in c.inputs.items():
        #         if isinstance(val, Symbol):
        #             if val.container.num is not None:
        #                 # if c.num is not None and val.container.num != c.num:
        #                 #     raise Error("Size mismatches: {} {}".format(
        #                 #         c.name, val.container.name))
        #                 dct[key] = np.zeros(val.container.num)
        #             else:
        #                 dct[key] = dtype(0.0)
        #         elif isinstance(val, Input):
        #             if val.num is not None:
        #                 if c.num is not None and val.num != c.num:
        #                     warn(
        #                         (
        #                             f"Size mismatches: [{c.name}:{c.num}] vs. [{val.name}: "
        #                             f"{val.num}]. Unless you are connecting Input object "
        #                             "directly to a Project container, this is likely a bug."
        #                         ),
        #                         err.NeuralNetworkWarning,
        #                     )
        #                 dct[key] = np.zeros(val.num, dtype=dtype)
        #             else:
        #                 dct[key] = dtype(0.0)
        #         elif isinstance(val, Number):
        #             dct[key] = dtype(val)
        #         else:
        #             raise err.NeuralNetworkCompileError(
        #                 f"Container wrapping [{c.obj}] input {key} value {val} not "
        #                 "understood"
        #             )

        #     if hasattr(c.obj, "compile"):
        #         try:
        #             if isinstance(c.obj, Model):
        #                 c.obj.compile(dtype=dtype, num=c.num, **dct)
        #             else:
        #                 c.obj.compile(**dct)
        #         except Exception as e:
        #             if debug:
        #                 s = "".join([", {}={}".format(*k) for k in dct.items()])
        #                 print(f"{c.name}.cuda_compile(dtype=dtype, num={c.num}{s})")
        #             raise err.NeuralNetworkCompileError(
        #                 f"Compilation Failed for Container {c.obj}"
        #             ) from e
        self._iscompiled = True

    def record(self, *args: tp.Iterable[Symbol]) -> None:
        """Record symbols (container.variables)"""
        for arg in args:
            if not isinstance(arg, Symbol):
                raise err.NeuralNetworkError(
                    f"{arg} needs to be an instance of Symbol."
                )
            arg.container.record(arg.key)
