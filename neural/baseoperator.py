"""Base Class of Operators
"""
from . import types as tpe
import numpy as np
import pycuda.gpuarray as garray


class Operator:
    backend: tpe.SupportedBackend = "numpy"

    def __init__(self, size=None, output_size=None, dtype=np.float64, backend="cuda"):
        self.size = size
        if output_size is None:
            output_size = size
        self._output_size = output_size
        self.dtype = "double" if dtype == np.float64 else "float"
        if self._output_size is not None:
            self.output = garray.empty(self._output_size, dtype=dtype)
        else:
            self.output = 0.0
        self._backend = backend
        if backend == "scalar":
            self.output = self.output.get()
        elif backend != "cuda":
            raise NotImplementedError("{} backend not understood".format(backend))

    def update(self, **kwargs):
        pass

    def compile(self, **kwargs):
        pass
