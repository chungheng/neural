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
import sys
import warnings
from collections import OrderedDict
from functools import reduce
from numbers import Number
from inspect import isclass
from warnings import warn
import typing as tp
import numpy as np
import pycuda.gpuarray as garray
from tqdm.auto import tqdm

# pylint:disable=relative-beyond-top-level
from ..basemodel import Model
from ..recorder import Recorder
from ..codegen.symbolic import SympyGenerator
from ..utils import MINIMUM_PNG
from .. import errors as err
# pylint:enable=relative-beyond-top-level

class Symbol(object):
    def __init__(
        self,
        container: tp.Any,
        key: str,  # DEBUG: container should be Container Type but it's only declared later
    ):
        self.container = container
        self.key = key

    def __getitem__(self, given: str) -> tp.Any:
        attr = getattr(self.container.recorder, self.key)
        return attr.__getitem__(given)


class Input(object):
    """Input Object for Neural Network

    Note:
        An Input object :code:`inp` can be updated using either of the 2 methods:

        1. :code:`value = next(inp)`
        2. :code:`inp.step(); value = inp.value`

        The latter method is useful if an input object's value is to be read by
        multiple containers.
    """

    def __init__(self, num: int = None, name: str = None):
        self.num = num
        self.name = name
        self.data = None
        self.steps = 0
        self.iter = None
        self.latex_src = "External stimulus"
        self.graph_src = MINIMUM_PNG
        self.value = None

    def __call__(self, data):
        if data.ndim == 1:
            if self.num != 1 and self.num is not None:
                raise err.NeuralNetworkInputError(
                    f"Input '{self.name}' is specified with num={self.num} but was "
                    f"given data of shape={data.shape}"
                )
        elif data.ndim == 2:
            if self.num != data.shape[1]:
                raise err.NeuralNetworkInputError(
                    f"Input '{self.name}' is specified with num={self.num} but was "
                    f"given data of shape={data.shape}"
                )
        else:
            raise err.NeuralNetworkInputError(
                f"Input '{self.name}' is given data of shape={data.shape}, only up-to "
                "2D data is supported currently."
            )
        self.data = data
        self.steps = len(data) if hasattr(data, "__len__") else 0
        self.iter = iter(self.data)
        if hasattr(data, "__iter__"):
            self.iter = iter(self.data)
        elif isinstance(data, garray.GPUArray):
            self.iter = (x for x in self.data)
        else:
            raise TypeError()

        return self

    def step(self):
        self.value = next(self)

    def __next__(self):
        return next(self.iter)

    def reset(self) -> None:
        if hasattr(self.data, "__iter__"):
            self.iter = iter(self.data)
        elif isinstance(self.data, garray.GPUArray):
            if not self.data.flags.c_contiguous:
                raise err.NeuralNetworkInputError(
                    f"Input {self.name} has non-contiguous pyCuda GPUArray as data, "
                    "need to be C-contiguous"
                )
            self.iter = (x for x in self.data)
        else:
            raise err.NeuralNetworkInputError(
                f"type of data {self.data} for input {self.name} not understood, "
                "need to be iterable or PyCuda.GPUArray"
            )


class Container(object):
    """
    A wrapper holds an Model instance with symbolic reference to its variables.

    Examples:

    .. code-block:: python

        >>> hhn = Container(HodgkinHuxley())
        >>> hhn.v  # reference to hhn.states['v']
    """

    def __init__(
        self, obj: tp.Union[Model, Symbol, Number, Input], num: int, name: str = None
    ):
        self.obj = obj
        self.num = num
        self.name = name or ""
        self.vars = dict()
        self.inputs = dict()
        self.recorder = None
        self._rec = []

    def __repr__(self):
        return f"Container[{self.obj}] - num {self.num}"

    def __call__(self, **kwargs):
        """Setup input connection to Container
        Notes:
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
                        f"Attempting to set variable '{key}' of container but the "
                        "variable is not understood"
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
        except Exception as e:
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
                msg = NeuralRecorderWarning(
                    f"Attribute {arg} not found in {self.obj}, skipping"
                )
                warn(msg)
                continue
            if arg not in self._rec:
                self._rec.append(arg)

    def set_recorder(
        self, steps: int, rate: int = 1, gpu_buffer: int = 500
    ) -> Recorder:
        """Create Recorder Instace

        Keyword Arguments:
            steps: total number of steps to record
            rate: sample rate at which the results are recorded
        """
        if not self._rec:
            self.recorder = None
        elif (
            (self.recorder is None)
            or (self.recorder.total_steps != steps)
            or (set(self.recorder.dct.keys()) != set(self._rec))
        ):
            self.recorder = Recorder(
                self.obj, self._rec, steps, gpu_buffer=gpu_buffer, rate=rate
            )
        return self.recorder

    def _get_latex(self) -> str:
        latex_src = f"{self.obj.__class__.__name__}:<br><br>"
        if isinstance(self.obj, Model):
            sg = SympyGenerator(self.obj)
            latex_src += sg.latex_src
            variables = [f"\({x}\)" for x in sg.signature]
            latex_src += "<br>Input: " + ", ".join(variables)
            variables = []
            for _k, _v in sg.variables.items():
                if (_v.type in ["state", "intermediate"]) and _v.integral == None:
                    variables.append(f"\({_k}\)")
            latex_src += "<br>Variables: " + ", ".join(variables)

        return latex_src

    def _get_graph(self):
        if isinstance(self.obj, Model):
            return self.obj.to_graph()
        return MINIMUM_PNG

    @classmethod
    def isacceptable(cls, module_or_obj) -> bool:
        """Check if a custom module or object is acceptable as Container"""
        return hasattr(module_or_obj, "update") and callable(module_or_obj.update)


class Network(object):
    """Neural Network Object"""

    def __init__(self, solver: str = "euler", backend: str = "cuda"):
        self.containers = OrderedDict()
        self.inputs = OrderedDict()
        self.solver = solver
        self.backend = backend
        self._iscompiled = False

    def input(self, num: int = None, name: str = None) -> Input:
        """Create input object"""
        name = name or f"input{len(self.inputs)}"
        inp = Input(num=num, name=name)
        self.inputs[name] = inp
        self._iscompiled = False
        return inp

    def add(
        self,
        module,
        num: int = None,
        name: str = None,
        record=None,
        solver=None,
        **kwargs,
    ):
        # backend = backend or self.backend
        solver = solver or self.solver
        name = name or f"obj{len(self.containers)}"
        if name in self.containers:
            raise err.NeuralNetworkError(
                f"Duplicate container name is not allowed: '{name}'"
            )

        record = record or []
        if isinstance(module, Model):
            obj = module
        elif issubclass(module, Model):
            obj = module(solver=solver, **kwargs)
        elif isclass(module):
            if not Container.isacceptable(module):
                raise err.NeuralNetworkError(
                    f"{module} is not an acceptable Container type"
                )
            kwargs["size"] = num
            obj = module(**kwargs)  # , backend=backend)
        else:
            raise err.NeuralNetworkError(
                f"{module} is not a submodule nor an instance of {Model}"
            )

        container = Container(obj, num, name)
        if record is not None:
            if isinstance(record, (tuple, list)):
                container.record(*record)
            else:
                container.record(record)

        self.containers[name] = container
        self._iscompiled = False
        return container

    def run(
        self,
        dt: float,
        steps: int = 0,
        rate: int = 1,
        verbose: str = None,
        **kwargs,
    ) -> None:
        """Run Network

        Keyword Arguments:
            dt: step size in second
            steps: number of steps to run, inferred from input if not specified
            rate: frequency of recording output
            verbose: Content to show for progressbar, set to `None`(default) to disable.
        """
        if not self._iscompiled:
            raise err.NeuralNetworkCompileError(
                "Please compile before running the network."
            )

        solver = kwargs.pop("solver", self.solver)
        gpu_buffer = kwargs.pop("gpu_buffer", 500)

        # calculate number of steps
        steps = reduce(max, [input.steps for input in self.inputs.values()], steps)
        for name, val in self.inputs.items():
            if val.steps == 0:
                warnings.warn(f"Input '{name}' has 0 steps", UserWarning)

        # create recorders
        for c in self.containers.values():
            recorder = c.set_recorder(steps, rate, gpu_buffer=gpu_buffer)

        # reset everything
        self.reset()

        # create iterator for simulation loop
        iterator = range(steps)
        if verbose is not None:
            if isinstance(verbose, str):
                iterator = tqdm(iterator, total=steps, desc=verbose, dynamic_ncols=True)
            else:
                iterator = tqdm(iterator, total=steps, dynamic_ncols=True)

        for i in iterator:
            self.update(dt, i)

    def reset(self) -> None:
        """Reset everything"""
        # reset recorders
        for c in self.containers.values():
            if c.recorder is not None:
                c.recorder.reset()

        # reset Model variables
        for c in self.containers.values():
            if isinstance(c.obj, Model):
                c.obj.reset()

        # reset inputs
        for input in self.inputs.values():
            input.reset()

    def update(self, dt: float, idx: int) -> None:
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
            args = {}
            for key, val in c.inputs.items():
                if isinstance(val, Symbol):
                    args[key] = getattr(val.container.obj, val.key)
                elif isinstance(val, Input):
                    args[key] = val.value
                elif isinstance(val, Number):
                    args[key] = val
                else:
                    raise err.NeuralNetworkUpdateError(
                        f"Input '{key}' of container '{c}' has invalid type: {val}"
                    )

            if isinstance(c.obj, Model):
                try:
                    c.obj.update(dt, **args)
                except Exception as e:
                    raise err.NeuralNetworkUpdateError(
                        f"Update Failed for Model [{c.obj}]"
                    ) from e
            else:
                try:
                    c.obj.update(**args)
                except Exception as e:
                    raise err.NeuralNetworkUpdateError(
                        f"Update Failed for Model [{c.obj}]"
                    ) from e
        # update recorders
        for c in self.containers.values():
            if c.recorder is not None:
                c.recorder.update(idx)

    def compile(
        self, dtype: tp.Any = None, debug: bool = False, backend: str = "cuda"
    ) -> None:
        backend = backend or self.backend
        dtype = dtype or np.float64
        for c in self.containers.values():
            dct = {}
            for key, val in c.inputs.items():
                if isinstance(val, Symbol):
                    if val.container.num is not None:
                        # if c.num is not None and val.container.num != c.num:
                        #     raise Error("Size mismatches: {} {}".format(
                        #         c.name, val.container.name))
                        dct[key] = np.zeros(val.container.num)
                    else:
                        dct[key] = dtype(0.0)
                elif isinstance(val, Input):
                    if val.num is not None:
                        if c.num is not None and val.num != c.num:
                            # raise Exception("Size mismatches: [{}: {}] vs. [{}: {}]".format(
                            #     c.name, c.num, val.name, val.num))
                            warn((
                                f"Size mismatches: [{c.name}:{c.num}] vs. [{val.name}: "
                                f"{val.num}]. Unless you are connecting Input object "
                                "directly to a Project container, this is likely a bug."
                                ), 
                                err.NeuralNetworkWarning
                            )
                        dct[key] = np.zeros(val.num)
                    else:
                        dct[key] = dtype(0.0)
                elif isinstance(val, Number):
                    dct[key] = dtype(val)
                else:
                    raise err.NeuralNetworkCompileError(
                        f"Container wrapping [{c.obj}] input {key} value {val} not "
                        "understood"
                    )

            if hasattr(c.obj, "compile"):
                try:
                    if isinstance(c.obj, Model):
                        c.obj.compile(backend=backend, dtype=dtype, num=c.num, **dct)
                    else:
                        c.obj.compile(**dct)
                except Exception as e:
                    raise Exception(f"Compilation Failed for Container {c.obj}") from e
                if debug:
                    s = "".join([", {}={}".format(*k) for k in dct.items()])
                    print(f"{c.name}.cuda_compile(dtype=dtype, num={c.num}{s})")
        self._iscompiled = True

    def record(self, *args: tp.Iterable[Symbol]):
        for arg in args:
            if not isinstance(arg, Symbol):
                raise err.NeuralNetworkError(
                    f"{arg} needs to be an instance of Symbol."
                )
            arg.container.record(arg.key)

    def get_obj(self, name: str) -> tp.Union[Container, Input]:
        if name in self.containers:
            return self.containers[name]
        elif name in self.inputs:
            return self.inputs[name]
        else:
            raise err.NeuralNetworkError(f"Unexpected name: '{name}'")

    def to_graph(self, png: bool = False, svg: bool = False, prog="dot"):
        """Visualize Network instance as Graph

        Arguments:
            network: network to visualize

        Keyword Arguments:
            png : whether to return png image as output
            svg : whether to return svg image as output
                - png takes precedence over svg
            prog: program used to optimize graph layout
        """
        try:
            import pydot
        except ImportError as e:
            raise err.NeuralNetworkError(
                "pydot needs to be installed to create graph from network"
            ) from e

        graph = pydot.Dot(
            graph_type="digraph", rankdir="LR", splines="ortho", decorate=True
        )

        nodes = {}
        for c in list(self.containers.values()) + list(self.inputs.values()):
            node = pydot.Node(c.name, shape="rect")
            nodes[c.name] = node
            graph.add_node(node)

        edges = []
        for c in self.containers.values():
            target = c.name
            v = nodes[target]
            for key, val in c.inputs.items():
                if isinstance(val, Symbol):
                    source = val.container.name
                    label = val.key
                elif isinstance(val, Input):
                    source = val.name
                    label = ""
                else:
                    raise err.NeuralNetworkError(
                        f"Container wrapping [{c.obj}] input {key} value {val} not "
                        "understood"
                    )
                u = nodes[source]
                graph.add_edge(pydot.Edge(u, v, label=label))
                edges.append((source, target, label))

        if png:  # return PNG Directly
            png_str = graph.create_png(prog="dot")  # pylint: disable=no-member
            return png_str
        elif svg:
            svg_str = graph.create_svg(prog="dot")  # pylint: disable=no-member
            return svg_str
        else:
            D_bytes = graph.create_dot(prog="dot")  # pylint: disable=no-member

            D = str(D_bytes, encoding="utf-8")

            if D == "":  # no data returned
                print(
                    f"""Graphviz layout with {prog} failed
                    To debug what happened try:
                    >>> P = nx.nx_pydot.to_pydot(G)
                    >>> P.write_dot("file.dot")
                    And then run {prog} on file.dot"""
                )

            # List of "pydot.Dot" instances deserialized from this string.
            Q_list = pydot.graph_from_dot_data(D)
            assert len(Q_list) == 1
            Q = Q_list[0]
            # return Q

            def get_node(Q, n):
                node = Q.get_node(n)

                if isinstance(node, list) and len(node) == 0:
                    node = Q.get_node('"{}"'.format(n))
                    assert node

                return node[0]

            def get_label_xy(x, y, ex, ey):
                min_dist = np.inf
                min_ex, min_ey = [0, 0], [0, 0]
                for _ex, _ey in zip(zip(ex, ex[1:]), zip(ey, ey[1:])):
                    dist = (np.mean(_ex) - x) ** 2 + (np.mean(_ey) - y) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        min_ex[:] = _ex[:]
                        min_ey[:] = _ey[:]
                if min_ex[0] == min_ex[1]:
                    _x = min_ex[0]
                    _x = np.sign(x - _x) * 10 + _x
                    _y = y
                else:
                    _x = x
                    _y = min_ey[0]
                    _y = np.sign(y - _y) * 10 + _y
                return _x, _y - 3

            elements = []
            bb = Q.get_bb()
            viewbox = bb[1:-1].replace(",", " ")

            for n in nodes.keys():

                node = get_node(Q, n)

                # strip leading and trailing double quotes
                pos = node.get_pos()[1:-1]

                if pos is not None:
                    obj = self.get_obj(n)
                    w = float(node.get_width())
                    h = float(node.get_height())

                    x, y = map(float, pos.split(","))
                    attrs = {
                        "width": w,
                        "height": h,
                        "rx": 5,
                        "ry": 5,
                        "x": x,
                        "y": y,
                        "stroke-width": 1.5,
                        "fill": "none",
                        "stroke": "#48caf9",
                    }

                    elements.append(
                        {
                            "label": [n, x, y],
                            "shape": "rect",
                            "attrs": attrs,
                            "latex": obj.latex_src,
                            "graph": obj.graph_src,
                        }
                    )

            min_x, min_y, scale_w, scale_h = np.inf, np.inf, 0, 0
            for el in elements:
                if min_x > el["attrs"]["x"]:
                    min_x = el["attrs"]["x"]
                    scale_w = 2 * min_x / el["attrs"]["width"]
                if min_y > el["attrs"]["y"]:
                    min_y = el["attrs"]["y"]
                    scale_h = 2 * min_y / el["attrs"]["height"]
            for el in elements:
                w = scale_w * el["attrs"]["width"]
                h = scale_h * el["attrs"]["height"]
                el["attrs"]["x"] = el["attrs"]["x"] - w / 2
                el["attrs"]["y"] = el["attrs"]["y"] - h / 2
                el["attrs"]["width"] = w
                el["attrs"]["height"] = h

            for e in Q.get_edge_list():
                pos = (e.get_pos()[1:-1]).split(" ")
                ax, ay = [float(v) for v in pos[0].split(",")[1:]]
                pos = [v.split(",") for v in pos[1:]]

                xx = [float(v[0]) for v in pos] + [ax]
                yy = [float(v[1]) for v in pos] + [ay]
                x, y, _x, _y = [], [], 0, 0
                for __x, __y in zip(xx, yy):
                    if not (__x == _x and __y == _y):
                        x.append(__x)
                        y.append(__y)
                    _x = __x
                    _y = __y
                path = [f"{_x} {_y}" for _x, _y in zip(x, y)]
                p = "M" + " L".join(path)
                attrs = {"d": p, "stroke-width": 1.5, "fill": "none", "stroke": "black"}
                lp = e.get_lp()
                if lp:
                    lx, ly = [float(v) for v in lp[1:-1].split(",")]
                    lx, ly = get_label_xy(lx, ly, x, y)
                    label = [e.get_label() or "", lx, ly]
                elements.append({"label": label, "shape": "path", "attrs": attrs})
            output = {"elements": elements, "viewbox": viewbox}
            return output
