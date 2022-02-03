"""Solvers for Model
"""
import collections
from .base_backend import Backend
from ._scipy import SciPyBackend
from ._numpy import NumPyBackend

BACKENDS = collections.namedtuple("Backends", "numpy scipy")(NumPyBackend, SciPyBackend)
