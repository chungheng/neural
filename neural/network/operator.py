import numpy as np
import pycuda
from pycuda.elementwise import ElementwiseKernel
import pycuda.gpuarray as garray
import skcuda
import skcuda.misc
import skcuda.linalg

class Sum(object):
    def __init__(self, dtype=np.float64):
        self.output = garray.empty(1, dtype=dtype)
    def update(self, input):
        skcuda.misc.sum(input, out=self.output)

class BlockSum(object):
    def __init__(self, size=1, block_size=1, dtype=np.float64):
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
