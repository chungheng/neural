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
    def __init__(self):
        pass

    def __call__(self):
        pass

class Network(object):
    """
    """
    def __init__(self):
        pass

    def add(self, module, **kwargs):
        pass

    def run(self):
        pass

    def compile(self):
        pass

    def record(self):
        pass
