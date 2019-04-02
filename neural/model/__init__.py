import inspect
import sys

from . import neuron
from . import synapse
from ..basemodel import Model

for module in (neuron, synapse):
    for name in dir(module):
        attr = getattr(module, name)

        if inspect.isclass(attr) and issubclass(attr, Model):
            setattr(sys.modules[__name__], name, attr)

del name, attr, inspect, sys, Model, synapse, neuron, module
