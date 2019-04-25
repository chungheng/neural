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
from ..codegen.symbolic import SympyGenerator
from ..utils import MINIMUM_PNG

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
        self.latex_src = "External stimulus"
        self.graph_src = MINIMUM_PNG

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
        self.latex_src = self._get_latex()
        self.graph_src = self._get_graph()
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

    def _get_latex(self):

        latex_src = "{}:<br><br>".format(self.obj.__class__.__name__)
        if isinstance(self.obj, Model):
            sg = SympyGenerator(self.obj)

            latex_src += sg.latex_src
            vars = ["\({}\)".format(x) for x in sg.signature]
            latex_src += "<br>Input: " + ", ".join(vars)
            vars = []
            for _k, _v in sg.variables.items():
                if (_v.type == 'state' or _v.type == 'intermediate') \
                    and (_v.integral == None):
                    vars.append("\({}\)".format(_k))

            latex_src += "<br>Variables: " + ", ".join(vars)

        return latex_src

    def _get_graph(self):
        if isinstance(self.obj, Model):
            return self.obj.to_graph()
        else:
            return MINIMUM_PNG


    @classmethod
    def isacceptable(cls, obj):
        return hasattr(obj, "update") and callable(obj.update)

class Network(object):
    """
    """
    def __init__(self):
        self.containers = OrderedDict()
        self.inputs = OrderedDict()

        self._iscompiled = False

    def input(self, num=None, name=None):
        name = name or "input{}".format(len(self.inputs))
        input = Input(num=num, name=name)
        self.inputs[name] = input
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
            obj = module(num, **kwargs)
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
        steps = reduce(max, [input.steps for input in self.inputs.values()], steps)

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
        for input in self.inputs.values():
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

    def to_graph(self, bqplot=False, svg=False):
        import pydot
        graph = pydot.Dot(graph_type='digraph', rankdir='LR', splines='ortho',
        decorate=True)

        nodes = {}
        for c in list(self.containers.values())+list(self.inputs.values()):
            node = pydot.Node(c.name, shape='rect')
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
                    dist = (np.mean(_ex) - x)**2 + (np.mean(_ey) - y)**2
                    if dist < min_dist:
                        min_dist = dist
                        min_ex[:] = _ex[:]
                        min_ey[:] = _ey[:]
                if min_ex[0] == min_ex[1]:
                    _x = min_ex[0]
                    _x = np.sign(x-_x)*10+_x
                    _y = y
                else:
                    _x = x
                    _y = min_ey[0]
                    _y = np.sign(y-_y)*10+_y
                return _x, _y-3

            elements =[]
            bb = Q.get_bb()
            viewbox = bb[1:-1].replace(',', ' ')

            for n in nodes.keys():

                node = get_node(Q, n)

                # strip leading and trailing double quotes
                pos = node.get_pos()[1:-1]

                if pos is not None:
                    w = float(node.get_width())
                    h = float(node.get_height())

                    x, y = map(float, pos.split(","))
                    attrs = {'width': w, 'height': h, 'rx':5, 'ry':5,
                        'x': x, 'y': y,
                        'fill': 'none', 'stroke': '#000000'}

                    if n in self.containers:
                        val = self.containers[n]
                        if isinstance(val.obj, Model):
                            sg = SympyGenerator(val.obj)
                            latex_src = [sg.latex_src, sg.signature, []]
                            for _k, _v in sg.variables.items():
                                if (_v.type == 'state' or _v.type == 'intermediate') \
                                    and (_v.integral == None):
                                    latex_src[2].append(_k)

                    elements.append({'label':[n, x, y], 'shape': 'rect', 'attrs':attrs, 'latex':latex_src})

            min_x, min_y, scale_w, scale_h = np.inf, np.inf, 0, 0
            for el in elements:
                if min_x > el['attrs']['x']:
                    min_x = el['attrs']['x']
                    scale_w = 2*min_x / el['attrs']['width']
                if min_y > el['attrs']['y']:
                    min_y = el['attrs']['y']
                    scale_h = 2*min_y / el['attrs']['height']
            for el in elements:
                w = scale_w * el['attrs']['width']
                h = scale_h * el['attrs']['height']
                el['attrs']['x'] = el['attrs']['x'] - w/2
                el['attrs']['y'] = el['attrs']['y'] - h/2
                el['attrs']['width'] = w
                el['attrs']['height'] = h

            for e in Q.get_edge_list():
                pos = (e.get_pos()[1:-1]).split(' ')
                ax, ay = [float(v) for v in pos[0].split(',')[1:]]
                pos = [v.split(',') for v in pos[1:]]

                xx = [float(v[0]) for v in pos] + [ax]
                yy = [float(v[1]) for v in pos] + [ay]
                x, y, _x, _y = [], [], 0, 0
                for __x, __y in zip(xx, yy):
                    if not (__x == _x and __y == _y):
                        x.append(__x)
                        y.append(__y)
                    _x = __x
                    _y = __y
                path = ["{} {}".format(_x, _y) for _x, _y in zip(x, y)]
                p = 'M' + " L".join(path)
                attrs = dict(d=p, stroke_width=1., fill='none', stroke='black')
                lp = e.get_lp()
                if lp:
                    lx, ly = [float(v) for v in lp[1:-1].split(',')]
                    lx, ly = get_label_xy(lx, ly, x, y)
                    label = [e.get_label() or '', lx, ly]
                elements.append({'label':label, 'shape': 'path', 'attrs':attrs})
            output = {'elements': elements, 'viewbox': viewbox}
            return output

    def get_obj(self, name):
        if name in self.containers:
            return self.containers[name]
        elif name in self.inputs:
            return self.inputs[name]
        else:
            raise TypeError("Unexpected name: '{}'".format(name))
