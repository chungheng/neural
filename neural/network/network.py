"""
Network module for constructing an abitrary circuit of neurons and synapses.

Examples:

>>> nn = Network()
>>> iaf = nn.add(IAF, bias=0., c=10., record=['v'])
>>> syn = nn.add(AlphaSynapse, ar=1e-1, stimulus=iaf.spike)
>>> hhn = nn.add(HodgkinHuxley, stimulus=syn.I)
>>> nn.input(s=iaf.stimulus)
>>>
>>> nn.compile(dtype=dtype)
>>>
>>> nn.run(dt, s=numpy.random.rand(10000))
"""
import sys
from collections import OrderedDict
from functools import reduce
from numbers import Number
from inspect import isclass
from warnings import warn
import numpy as np
import pycuda.gpuarray as garray
from tqdm import tqdm
import typing as tp

from ..basemodel import Model
from ..recorder import Recorder
from ..codegen.symbolic import SympyGenerator
from ..utils import MINIMUM_PNG
from ..logger import (
    NeuralNetworkError,
    NeuralNetworkWarning,
    NeuralNetworkCompileError,
    NeuralNetworkUpdateError,
    NeuralNetworkInputError,
)

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

if PY2:
    raise NeuralNetworkError("neural.network does not support Python 2.")


class Symbol(object):
    def __init__(self, container: Container, key: str):
        self.container = container
        self.key = key

    def __getitem__(self, given: str) -> tp.Any:
        attr = getattr(self.container.recorder, self.key)
        return attr.__getitem__(given)


class Input(object):
    """Input Object for Neural Network

    Note:
        An Input object `inp` can be updated using either of the 2 methods:
            1. `value = next(inp)`
            2. `inp.step(); value = inp.value`
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
        if (self.num is None and data.ndim > 1) or (
            data.ndim == 2 and data.shape[1] != self.num
        ):
            raise NeuralNetworkInputError(
                f"Input {self.name} is specified with num {self.num} but was given data of shape {data.shape}"
            )
        if data.ndim > 2:
            raise NeuralNetworkInputError(
                f"Input {self.name} is given data of shape {data.shape}, only up-to 2D data is supported currently."
            )
        self.data = data
        self.steps = len(data) if hasattr(data, "__len__") else 0
        self.reset()  # create iter object
        return self

    def step(self) -> None:
        self.value = next(self)

    def __next__(self):
        return next(self.iter)

    def reset(self) -> None:
        if hasattr(self.data, "__iter__"):
            self.iter = iter(self.data)
        elif isinstance(self.data, garray.GPUArray):
            if not self.data.flags.c_contiguous:
                raise NeuralNetworkInputError(
                    f"Input {self.name} has non-contiguous pyCuda GPUArray as data, need to be C-contiguous"
                )
            self.iter = (x for x in self.data)
        else:
            raise NeuralNetworkInputError(
                f"type of data {self.data} for input {self.name} not understood, need to be iterable or PyCuda.GPUArray"
            )


class Container(object):
    """
    A wrapper holds an Model instance with symbolic reference to its varaibles.

    Examples:
    >>> hhn = Container(HodgkinHuxley)
    >>> hhn.v  # reference to hhn.states['v']
    """

    def __init__(self, obj, num, name=None):
        self.obj = obj
        self.num = num
        self.name = name or ""
        self.vars = {}
        self.inputs = dict()
        self.recorder = None
        self._rec = []

    def __call__(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(self.obj, Model):
                if key in self.obj.Variables:
                    setattr(self.obj, key, val)
                elif key in self.obj.Inputs:
                    assert isinstance(val, (Symbol, Number, Input))
                    self.inputs[key] = val
                else:
                    raise KeyError("Unexpected variable '{}'".format(key))
            else:
                assert isinstance(val, (Symbol, Number, Input))
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

    def record(self, *args) -> None:
        for arg in args:
            _ = getattr(self.obj, arg)
            if arg not in self._rec:
                self._rec.append(arg)

    def set_recorder(self, steps: int, rate: int = 1) -> Recorder:
        if not self._rec:
            self.recorder = None
        elif (
            (self.recorder is None)
            or (self.recorder.total_steps != steps)
            or (set(self.recorder.dct.keys()) != set(self._rec))
        ):
            self.recorder = Recorder(
                self.obj, self._rec, steps, gpu_buffer=500, rate=rate
            )
        return self.recorder

    def _get_latex(self) -> str:
        latex_src = f"{self.obj.__class__.__name__}:<br><br>"
        if isinstance(self.obj, Model):
            sg = SympyGenerator(self.obj)
            latex_src += sg.latex_src
            vars = [f"\({x}\)" for x in sg.signature]
            latex_src += "<br>Input: " + ", ".join(vars)
            vars = []
            for _k, _v in sg.variables.items():
                if (_v.type == "state" or _v.type == "intermediate") and (
                    _v.integral == None
                ):
                    vars.append(f"\({_k}\)")
            latex_src += "<br>Variables: " + ", ".join(vars)

        return latex_src

    def _get_graph(self):
        if isinstance(self.obj, Model):
            return self.obj.to_graph()
        return MINIMUM_PNG

    @classmethod
    def isacceptable(cls, obj) -> bool:
        """Check if a custom module is acceptable as Container"""
        return hasattr(obj, "update") and callable(obj.update)


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
        input = Input(num=num, name=name)
        self.inputs[name] = input
        self._iscompiled = False
        return input

    def add(
        self,
        module,
        num: int = None,
        name: str = None,
        record=None,
        # backend=None,
        solver=None,
        **kwargs,
    ):
        # backend = backend or self.backend
        solver = solver or self.solver
        name = name or f"obj{len(self.containers)}"
        record = record or []
        if isinstance(module, Model):
            obj = module
        elif issubclass(module, Model):
            obj = module(solver=solver, **kwargs)
        elif isclass(module):
            if not Container.isacceptable(module):
                raise NeuralNetworkError(
                    f"{module} is not an acceptable Container type"
                )
            kwargs["size"] = num
            obj = module(**kwargs, backend=backend)
        else:
            raise NeuralNetworkError(
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
        verbose: bool = False,
        solver: str = None,
        session_name: str = "Network",
        **kwargs,
    ) -> None:
        """Run Network

        Keyword Arguments:
            dt: step size in second
            steps: number of steps to run, inferred from input if not specified
            rate: frequency of recording output
            verbose: whether to show progress
            solver: specify solver to use
            session_name: name of the running session name, reflected in progress bar
        """
        if not self._iscompiled:
            raise NeuralNetworkCompileError(
                "Please compile before running the network."
            )
        solver = solver or self.solver

        # calculate number of steps
        steps = reduce(max, [input.steps for input in self.inputs.values()], steps)

        # reset recorders
        recorders = []
        for c in self.containers.values():
            recorder = c.set_recorder(steps, rate)
            if recorder is not None:
                recorders.append(recorder)

        # reset Modle variables
        for c in self.containers.values():
            if isinstance(c.obj, Model):
                c.obj.reset()

        # reset inputs
        for input in self.inputs.values():
            input.reset()

        iterator = range(steps)
        if verbose:
            iterator = tqdm(iterator, total=steps, desc=session_name)

        for i in iterator:
            for c in self.inputs.values():  # 1. update input
                c.step()

            for c in self.containers.values():  # 2. update containers
                args = {}
                for key, val in c.inputs.items():
                    if isinstance(val, Symbol):
                        args[key] = getattr(val.container.obj, val.key)
                    elif isinstance(val, Input):
                        args[key] = val.value
                    elif isinstance(val, Number):
                        args[key] = val
                    else:
                        raise NeuralNetworkCompileError(
                            f"Container wrapping [{c.obj}] input {key} value {val} not understood"
                        )
                if isinstance(c.obj, Model):
                    try:
                        c.obj.update(dt, **args)
                    except Exception as e:
                        raise NeuralNetworkUpdateError(
                            f"Container wrapping [{c.obj}] Error"
                        ) from e
                else:
                    try:
                        c.obj.update(**args)
                    except Exception as e:
                        raise NeuralNetworkUpdateError(
                            f"Container wrapping [{c.obj}] Error"
                        ) from e
            for recorder in recorders:  # 3. update recorder
                recorder.update(i)

    def compile(
        self, dtype: tp.Any = None, debug: bool = False, backend: str = None
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
                            err = NeuralNetworkWarning(
                                f"""Size mismatches: [{c.name}: {c.num}] vs. [{val.name}: {val.num}].
                                Unless you are connecting Input object directly to a Project container,
                                this is likely a bug.
                                """
                            )
                            warn(err)
                        dct[key] = np.zeros(val.num)
                    else:
                        dct[key] = dtype(0.0)
                elif isinstance(val, Number):
                    dct[key] = dtype(val)
                else:
                    raise NeuralNetworkCompileError(
                        f"Container wrapping [{c.obj}] input {key} value {val} not understood"
                    )

            if hasattr(c.obj, "compile"):
                if isinstance(c.obj, Model):
                    c.obj.compile(backend=backend, dtype=dtype, num=c.num, **dct)
                else:
                    c.obj.compile(**dct)
                if debug:
                    s = "".join([", {}={}".format(*k) for k in dct.items()])
                    print(f"{c.name}.cuda_compile(dtype=dtype, num={c.num}{s})")
        self._iscompiled = True

    def record(self, *args: tp.Iterable[Symbol]):
        for arg in args:
            if not isinstance(arg, Symbol):
                raise NeuralNetworkError(f"{arg} needs to be an instance of Symbol.")
            arg.container.record(arg.key)

    def get_obj(self, name: str) -> tp.Union[Container, Input]:
        if name in self.containers:
            return self.containers[name]
        elif name in self.inputs:
            return self.inputs[name]
        else:
            raise NeuralNetworkError(f"Unexpected name: '{name}'")

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
            raise NeuralNetworkError(
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
                    raise NeuralNetworkError(
                        f"Container wrapping [{c.obj}] input {key} value {val} not understood"
                    )
                u = nodes[source]
                graph.add_edge(pydot.Edge(u, v, label=label))
                edges.append((source, target, label))

        if png:  # return PNG Directly
            png_str = graph.create_png(prog="dot")
            return png_str
        elif svg:
            svg_str = graph.create_svg(prog="dot")
            return svg_str
        else:
            D_bytes = graph.create_dot(prog="dot")

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
