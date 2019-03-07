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
from numbers import Number

import numpy as np

from .basemodel import SimpleNamespace, Model

class Symbol(object):
    def __init__(self, container, key):
        self.container = container
        self.key = key

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
        self.vars = {key: Symbol(self, key) for key in obj.Variables.keys()}
        self.inputs = dict()

    def __call__(self, **kwargs):
        for key, val in kwargs.items():
            assert isinstance(val, (Symbol, Number))
            self.inputs[key] = val
        return self

    def __getattr__(self, key):
        if key in self.vars:
            return self.vars[key]

        return super(Container, self).__getattribute__(key)


class Network(object):
    """
    """
    def __init__(self):
        self.containers = []
        self.args = {}
        pass

    def input(self):
        pass

    def add(self, module, **kwargs):
        num = kwargs.pop('num', None)
        name = kwargs.pop('name', 'obj{}'.format(len(self.containers)))
        record = kwargs.pop('record', [])
        if isinstance(module, Model):
            obj = module
        elif issubclass(module, Model):
            obj = module(**kwargs)
        else:
            msg = "{} is not a submodule nor an instance of {}"
            raise ValueError(msg.format(module, Model))

        container = Container(obj, num, name)
        self.containers.append(container)

        return container

    def run(self):
        pass

    def compile(self, dtype=None):
        dtype = dtype or np.float64
        for c in self.containers:
            dct = {}
            for key, val in c.inputs.items():
                if isinstance(val, Symbol):
                    if val.container.num is not None:
                        assert val.container.num == c.num, "Size mismatches"
                    else:
                        dct[key] = dtype(0.)
                elif isinstance(val, Number):
                    dct[key] = dtype(val)
                else:
                    raise

            if hasattr(c.obj, 'cuda_compile'):
                c.obj.cuda_compile(dtype=dtype, num=c.num, **dct)
            _dct = "".join([", {}={}".format(k, v) for k, v in dct.items()])

        for c in self.containers:
            args = {}
            for key, val in c.inputs.items():
                if isinstance(val, Symbol):
                    args[key] = getattr(val.container.obj, val.key)
                elif isinstance(val, Number):
                    args[key] = val
                else:
                    raise

            self.args[c] = args

    def record(self):
        pass
