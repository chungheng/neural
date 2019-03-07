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
from inspect import isclass

import numpy as np
from tqdm import tqdm

from .basemodel import SimpleNamespace, Model

class Symbol(object):
    def __init__(self, container, key):
        self.container = container
        self.key = key

class Input(object):
    def __init__(self, num=None, name=None):
        self.num = num
        self.name = name

    def __call__(self, data):
        assert hasattr(data, '__iter__')
        self.data = data
        self.steps = len(data)
        self.iter = iter(self.data)

        return self

    def __next__(self):
        return next(self.iter)


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

    def __call__(self, **kwargs):
        for key, val in kwargs.items():
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

    @classmethod
    def isacceptable(cls, obj):
        return hasattr(obj, "update") and callable(obj.update)

class Network(object):
    """
    """
    def __init__(self):
        self.containers = []
        self.inputs = []

    def input(self, num=None, name=None):
        name = name or "obj{}".format(len(self.inputs))
        input = Input(num=num, name=name)
        self.inputs.append(input)
        return input

    def add(self, module, num=None, name=None, **kwargs):
        num = num
        name = name or "obj{}".format(len(self.containers))
        record = kwargs.pop('record', [])
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
        self.containers.append(container)

        return container

    def run(self, dt, steps):
        for i in tqdm(range(steps)):
            for c in self.containers:
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
                try:
                    c.obj.update(dt, **args)
                except:
                    print(c.name)

    def compile(self, dtype=None, debug=False):
        dtype = dtype or np.float64
        for c in self.containers:
            dct = {}
            for key, val in c.inputs.items():
                if isinstance(val, Symbol):
                    if val.container.num is not None:
                        if c.num is not None and val.container.num != c.num:
                            raise Error("Size mismatches: {} {}".format(
                                c.name, val.container.name))
                    else:
                        dct[key] = dtype(0.)
                elif isinstance(val, Input):
                    if val.num is not None:
                        if c.num is not None and val.num != c.num:
                            raise Error("Size mismatches: {} {}".format(
                                c.name, val.name))
                    else:
                        dct[key] = dtype(0.)
                elif isinstance(val, Number):
                    dct[key] = dtype(val)
                else:
                    raise

            if hasattr(c.obj, 'cuda_compile'):
                c.obj.cuda_compile(dtype=dtype, num=c.num, **dct)
                if debug:
                    s = ''.join([", {}={}".format(*k) for k in dct.items()])
                    print("{}.cuda_compile(dtype=dtype, num={}{})".format(
                    c.name, c.num, s))

    def record(self):
        pass
