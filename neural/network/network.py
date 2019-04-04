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
from collections import OrderedDict
from functools import reduce
from numbers import Number
from inspect import isclass
import sys

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

if PY2:
    raise Error("neural.network does not support Python 2.")

import numpy as np
from tqdm import tqdm

from ..basemodel import Model
from ..future import SimpleNamespace
from ..recorder import Recorder

class Symbol(object):
    def __init__(self, container, key):
        self.container = container
        self.key = key

    def __getitem__(self, given):
        attr = getattr(self.container.recorder, self.key)
        return attr.__getitem__(given)

class Input(object):
    def __init__(self, num=None, name=None):
        self.num = num
        self.name = name
        self.data = None
        self.steps = 0
        self.iter = None

    def __call__(self, data):
        assert hasattr(data, '__iter__')
        self.data = data
        self.steps = len(data) if hasattr(data, "__len__") else 0
        self.iter = iter(self.data)

        return self

    def __next__(self):
        return next(self.iter)

    def reset(self):
        self.iter = iter(self.data)

class Container(object):
    """
    A wrapper holds an Model instance with symbolic reference to its varaibles.

    Examples:
    >>> hhn = Container(HodgkinHuxley)
    >>> hhn.v # reference to hhn.states['v']
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
                if (key in self.obj.Variables):
                    setattr(self.obj, key, val)
                elif (key in self.obj.Inputs):
                    assert isinstance(val, (Symbol, Number, Input))
                    self.inputs[key] = val
                else:
                    raise KeyError("Unexpected variable '{}'".format(key))
            else:
                assert isinstance(val, (Symbol, Number, Input))
                self.inputs[key] = val

        return self

    def __getattr__(self, key):
        if key in self.vars:
            return self.vars[key]
        try:
            _ = getattr(self.obj, key)
            self.vars[key] = Symbol(self, key)
            return self.vars[key]
        except:
            return super(Container, self).__getattribute__(key)

    def record(self, *args):
        for arg in args:
            _ = getattr(self.obj, arg)
            if arg not in self._rec:
                self._rec.append(arg)

    def set_recorder(self, steps):
        if not self._rec:
            self.recorder = None
        elif (self.recorder is None) or \
            (self.recorder.total_steps != steps) or \
            (set(self.recorder.dct.keys()) != set(self._rec)):
            self.recorder = Recorder(self.obj, self._rec, steps, gpu_buffer=500)
        return self.recorder


    @classmethod
    def isacceptable(cls, obj):
        return hasattr(obj, "update") and callable(obj.update)

class Network(object):
    """
    """
    def __init__(self):
        self.containers = OrderedDict()
        self.inputs = []

        self._iscompiled = False

    def input(self, num=None, name=None):
        name = name or "obj{}".format(len(self.inputs))
        input = Input(num=num, name=name)
        self.inputs.append(input)
        self._iscompiled = False
        return input

    def add(self, module, num=None, name=None, record=None, **kwargs):
        num = num
        name = name or "obj{}".format(len(self.containers))
        record = record or []
        if isinstance(module, Model):
            obj = module
        elif issubclass(module, Model):
            obj = module(**kwargs)
        elif isclass(module):
            assert Container.isacceptable(module)
            obj = module(**kwargs)
        else:
            msg = "{} is not a submodule nor an instance of {}"
            raise ValueError(msg.format(module, Model))

        container = Container(obj, num, name)
        if record is not None:
            if isinstance(record, (tuple, list)):
                container.record(*record)
            else:
                container.record(record)

        self.containers[name] = container
        self._iscompiled = False
        return container

    def run(self, dt, steps=0, verbose=False):
        if not self._iscompiled:
            raise Error("Please compile before running the network.")

        # calculate number of steps
        steps = reduce(max, [input.steps for input in self.inputs], steps)

        # reset recorders
        recorders = []
        for c in self.containers.values():
            recorder = c.set_recorder(steps)
            if recorder is not None:
                recorders.append(recorder)

        # reset Modle variables
        for c in self.containers.values():
            if isinstance(c.obj, Model):
                c.obj.reset()

        # reset inputs
        for input in self.inputs:
            input.reset()

        iterator = range(steps)
        if verbose:
            iterator = tqdm(iterator)

        for i in iterator:
            for c in self.containers.values():
                args = {}
                for key, val in c.inputs.items():
                    if isinstance(val, Symbol):
                        args[key] = getattr(val.container.obj, val.key)
                    elif isinstance(val, Input):
                        args[key] = next(val)
                    elif isinstance(val, Number):
                        args[key] = val
                    else:
                        raise
                if isinstance(c.obj, Model):
                    c.obj.update(dt, **args)
                else:
                    c.obj.update(**args)
            for recorder in recorders:
                recorder.update(i)

    def compile(self, dtype=None, debug=False):
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
                        dct[key] = dtype(0.)
                elif isinstance(val, Input):
                    if val.num is not None:
                        if c.num is not None and val.num != c.num:
                            raise Error("Size mismatches: {} {}".format(
                                c.name, val.name))
                        dct[key] = np.zeros(val.num)
                    else:
                        dct[key] = dtype(0.)
                elif isinstance(val, Number):
                    dct[key] = dtype(val)
                else:
                    raise

            if hasattr(c.obj, 'compile'):
                if isinstance(c.obj, Model):
                    c.obj.compile(backend='cuda', dtype=dtype, num=c.num, **dct)
                else:
                    c.obj.compile(**dct)
                if debug:
                    s = ''.join([", {}={}".format(*k) for k in dct.items()])
                    print("{}.cuda_compile(dtype=dtype, num={}{})".format(
                    c.name, c.num, s))

        self._iscompiled = True

    def record(self, *args):
        for arg in args:
            assert isinstance(arg, Symbol)
            arg.container.record(arg.key)

    def to_graph(self, bqplot=False):
        import pydot
        graph = pydot.Dot(graph_type='digraph', rankdir='LR')

        nodes = {}
        for c in list(self.containers.values())+self.inputs:
            node = pydot.Node(c.name)
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
                    label = ''
                else:
                    raise
                u = nodes[source]
                graph.add_edge(pydot.Edge(u, v, label=label))
                edges.append((source, target, label))
        if not bqplot:
            png_str = graph.create_png(prog='dot')

            return png_str
        else:
            D_bytes = graph.create_dot(prog='dot')

            D = str(D_bytes, encoding='utf-8')
            scale = 50.

            if D == "":  # no data returned
                print("Graphviz layout with %s failed" % (prog))
                print()
                print("To debug what happened try:")
                print("P = nx.nx_pydot.to_pydot(G)")
                print("P.write_dot(\"file.dot\")")
                print("And then run %s on file.dot" % (prog))


            # List of "pydot.Dot" instances deserialized from this string.
            Q_list = pydot.graph_from_dot_data(D)
            assert len(Q_list) == 1
            Q = Q_list[0]

            node_x = []
            node_y = []
            node_data = []
            for n in nodes.keys():

                node = Q.get_node(n)

                if isinstance(node, list) and len(node) == 0:
                    node = Q.get_node('"{}"'.format(n))
                    assert node

                node = node[0]
                # strip leading and trailing double quotes
                pos = node.get_pos()[1:-1]
                if pos is not None:
                    w = float(node.get_width())*scale
                    h = float(node.get_height())*scale
                    attrs = {'rx': w/10., 'ry': h/10., 'width': w, 'height': h}
                    node_data.append({'label': n, 'shape': 'rect', 'id': n,
                        'shape_attrs': attrs})

                    x, y = pos.split(",")
                    node_x.append(float(x))
                    node_y.append(float(y))

            node2id = {n:i for i, n in enumerate(nodes.keys())}
            edges = [(node2id[u], node2id[v], l) for u, v, l in edges]
            link_data = [{'source': u, 'target': v} for u, v, l in edges]

            from bqplot import LinearScale
            from bqplot.marks import Graph
            xs = LinearScale()
            ys = LinearScale()
            bq_graph = Graph(
                node_data=node_data,
                link_data=link_data,
                link_type='line',
                x=node_x, y=node_y,
                colors=['#7DD9FA'] * len(nodes),
                scales={'x': xs, 'y': ys, },
                directed=True)

            return bq_graph
