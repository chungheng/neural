import numpy as np
import pycuda
from pycuda.elementwise import ElementwiseKernel
import pycuda.gpuarray as garray
import skcuda
import skcuda.misc
import skcuda.linalg

class Aggregate(object):
    def __init__(self, size, dtype=np.float64):
        self.output = garray.empty(size, dtype=dtype)
        self.dtype = 'double' if dtype == np.float64 else 'float'
    def update(self, **input):
        args = input.values()
        self._update(self.output, *args)
    def compile(self, **kwargs):
        num = len(kwargs)
        ins = ["in{}".format(i) for i in range(num)]
        self._update = ElementwiseKernel(
            ", ".join("{} *{}".format(self.dtype, x) for x in ['out'] + ins),
            "out[i] = {};".format(" + ".join(["{}[i]".format(x) for x in ins])),
            "aggregate")

class Sum(object):
    def __init__(self, dtype=np.float64):
        pass
    def update(self, input):
        self.output = skcuda.misc.sum(input)
        # print(skcuda.misc.sum(input), skcuda.misc.sum(input).dtype)

class BlockSum(object):
    def __init__(self, size, block_size=1, dtype=np.float64):
        self.block_size = block_size
        self.size = size
        self.output = garray.empty(int(size//block_size), dtype=dtype)
    def update(self, input):
        _input = input.reshape(-1, self.block_size)
        skcuda.misc.sum(_input, out=self.output, axis=1)

class Mean(object):
    def __init__(self, dtype=np.float64):
        self.output = garray.empty(1, dtype=dtype)
    def update(self, input):
        skcuda.misc.sum(input, out=self.output)

class BlockMean(object):
    def __init__(self, size=1, block_size=1, dtype=np.float64):
        self.block_size = block_size
        self.size = size
        self.output = garray.empty(int(size//block_size), dtype=dtype)
    def update(self, input):
        _input = input.reshape(-1, self.block_size)
        skcuda.misc.sum(_input, out=self.output, axis=1)

class Multiply(object):
    def __init__(self, multiplier, dtype=np.float64):
        if isinstance(multiplier, np.ndarray):
            multiplier = multiplier.astype(dtype)
            self.multiplier = garray.to_gpu(multiplier)
        elif isinstance(multiplier, garray.GPUArray):
            self.multiplier = multiplier
        else:
            raise TypeError("Unexpected type of multiplier.")

        self.output = garray.empty(multiplier.shape[0], dtype=dtype)
        self._output = self.output.reshape(-1, 1)

    def update(self, input):
        _input = input.reshape(-1, 1)
        skcuda.linalg.dot(self.multiplier, input, out=self._output)
