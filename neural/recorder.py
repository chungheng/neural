"""
Utility modules for recording data from Neural models
"""
from abc import abstractmethod
from numbers import Number

import sys
import time
import numpy as np

import pycuda
import pycuda.gpuarray as garray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype

from neural import INITIALIZED, CUDA, neural_initialized

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3


src_cuda = """
__global__ void copy(
    int num,
    int idx,
    int len,
    %(type)s *dst,
    %(type)s *src
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = tid; i < num; i += total_threads) {
        idx += len * i;
        dst[idx] = src[i];
    }
    return;
}
"""

_copy = {"float": None, "double": None}

if INITIALIZED and CUDA:
    for key, val in _copy.items():
        mod = SourceModule(src_cuda % {"type": key}, options=["--ptxas-options=-v"])
        func = mod.get_function("copy")
        func.prepare("iiiPP")
        _copy[key] = func


class Recorder(object):
    """
    Base recorder module.

    Attributes:
    """

    def __new__(cls, obj, attrs, steps, **kwargs):
        if cls is Recorder:
            attr = getattr(obj, attrs[0])
            if isinstance(attr, Number):
                return super(Recorder, cls).__new__(ScalarRecorder)
            elif isinstance(attr, np.ndarray):
                return super(Recorder, cls).__new__(NumpyRecorder)
            elif isinstance(attr, garray.GPUArray):
                return super(Recorder, cls).__new__(CUDARecorder)
            else:
                raise TypeError("{} of type: {}".format(attr, type(attr)))
        return super(Recorder, cls).__new__(cls)

    def __init__(self, obj, attrs, steps, **kwargs):
        self.obj = obj
        self.total_steps = steps
        self.rate = kwargs.pop("rate", 1)
        self.steps = int(steps / self.rate)

        self.dct = {key: None for key in attrs}

        # handle spike
        self._spike_recorder = dict()
        for atr in attrs:
            if "spike" in atr.lower():
                self._spike_recorder[atr] = 0
        self.init_dct()

        callback = kwargs.pop("callback", False)
        if callback:
            self.iter = iter(self)
            self.obj.add_callback(self)

    def __call__(self):
        return next(self.iter)

    def reset(self):
        self.iter = iter(self)

    @abstractmethod
    def init_dct(self):
        """
        initialize the dict that contains the numpy arrays
        """
        pass

    @abstractmethod
    def update(self, index):
        """
        update the record
        """
        pass

    def __iter__(self):
        for i in range(self.total_steps):
            if i % self.rate == 0:
                self.update(i)
            yield i

    def __getitem__(self, key):
        return self.dct[key]

    def __getattr__(self, key):
        if key in self.dct:
            return self.dct[key]
        return super(Recorder, self).__getattribute__(key)


class ScalarRecorder(Recorder):
    """
    Recorder for scalar data.

    Attributes:
    """

    def init_dct(self):
        for key in self.dct.keys():
            src = getattr(self.obj, key)
            assert isinstance(src, Number)
            self.dct[key] = np.zeros(self.steps, dtype=type(src))

    def update(self, index):
        d_index = int(index / self.rate)  # downsample index
        for key in self.dct.keys():
            self.dct[key][d_index] = getattr(self.obj, key)


class NumpyRecorder(Recorder):
    """
    Recorder for reading Numpy arrays of Neural models.

    Attributes:
    """

    def init_dct(self):
        for key in self.dct.keys():
            src = getattr(self.obj, key)
            assert isinstance(src, np.ndarray)
            shape = (src.size, self.steps)
            self.dct[key] = np.zeros(shape, order="F", dtype=src.dtype)

    def update(self, index):
        d_index = int(index / self.rate)  # downsample index
        for key in self.dct.keys():
            self.dct[key][:, d_index] = getattr(self.obj, key)


class CUDARecorder(Recorder):
    """
    Recorder for reading CUDA arrays of Neural models.

    Attributes:
    """

    def __init__(self, obj, attrs, steps, **kwargs):

        super(CUDARecorder, self).__init__(obj, attrs, steps, **kwargs)

        gpu_buffer = kwargs.pop("gpu_buffer", False)
        if gpu_buffer:
            self.buffer_length = self._get_buffer_length(gpu_buffer)
            self.gpu_dct = {}
            for key in attrs:
                src = getattr(self.obj, key)
                shape = (self.buffer_length, src.size)
                self.gpu_dct[key] = garray.zeros(shape, dtype=src.dtype)
            self._update = self._copy_memory_dtod
        else:
            self._update = self._copy_memory_dtoh

        if PY2:
            self.get_buffer = self._py2_get_buffer
        if PY3:
            self.get_buffer = self._py3_get_buffer

    def init_dct(self):
        for key in self.dct.keys():
            src = getattr(self.obj, key)
            assert isinstance(src, garray.GPUArray)
            shape = (src.size, self.steps)
            self.dct[key] = np.zeros(shape, order="F", dtype=src.dtype)

    def update(self, index):
        # update spike counter if has spike attributes
        if self._spike_recorder:
            for key in self._spike_recorder.keys():
                self._spike_recorder[key] += getattr(self.obj, key).get()

        if (index % self.rate) != 0:
            return
        self._update(index)
        for key in self._spike_recorder.keys():
            self._spike_recorder[key] = 0

    def _get_buffer_length(self, gpu_buffer):
        if gpu_buffer == "full" or gpu_buffer == "whole" or gpu_buffer is True:
            return self.steps
        else:
            return min(gpu_buffer, self.steps)

    def _copy_memory_dtod(self, index):
        # downsample index
        d_index = int(index / self.rate)
        # buffer index
        b_index = d_index % self.buffer_length
        for key in self.dct.keys():
            if key in self._spike_recorder:
                self.dct[key][:, d_index] = self._spike_recorder[key]
                continue
            src = getattr(self.obj, key)
            dst = int(self.gpu_dct[key].gpudata) + b_index * src.nbytes
            cuda.memcpy_dtod(dst, src.gpudata, src.nbytes)

        if (d_index == self.steps - 1) or (b_index == self.buffer_length - 1):
            for key in self.dct.keys():
                if key in self._spike_recorder:
                    continue
                else:
                    buffer = self.get_buffer(key, d_index)
                    cuda.memcpy_dtoh(buffer, self.gpu_dct[key].gpudata)

    def _copy_memory_dtoh(self, index):
        d_index = int(index / self.rate)  # downsample index
        for key in self.dct.keys():
            if key in self._spike_recorder:
                self.dct[key][:, d_index] = self._spike_recorder[key]
            else:
                self.dct[key][:, d_index] = getattr(self.obj, key).get()

    def _py2_get_buffer(self, key, index):
        beg = int(index / self.buffer_length) * self.buffer_length
        nbytes = self.gpu_dct[key].nbytes / self.buffer_length
        offset = int(beg * nbytes)
        size = int((index - beg + 1) * nbytes)
        return np.getbuffer(self.dct[key], offset, size)

    def _py3_get_buffer(self, key, index):
        mv = memoryview(self.dct[key].T)
        beg = int(index / self.buffer_length) * self.buffer_length
        return mv[beg : index + 1]
