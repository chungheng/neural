# pylint:disable=no-member
"""
Utility modules for recording data from Neural models
"""
import sys
from abc import abstractmethod
from numbers import Number
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.gpuarray as garray
import pycuda.driver as cuda

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3


class Recorder(object):
    """
    Base recorder module.

    Attributes:
    """

    def __new__(cls, obj, attrs, steps, rate=1, **kwargs):
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

    def __init__(self, obj, attrs, steps, rate=1, **kwargs):
        self.obj = obj
        self.total_steps = steps
        self.rate = rate
        self.steps = len(np.arange(steps)[::rate])

        self.dct = {key: None for key in attrs}
        self.init_dct()

        # get spike_variables
        self.spike_vars = tuple([key for key in attrs if "spike" in key.lower()])

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

    @abstractmethod
    def update(self, index):
        """
        update the record
        """

    def __iter__(self):
        for i in range(self.total_steps):
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

        # increment spike count directly in dct
        for key in self.spike_vars:
            self.dct[key][d_index] += getattr(self.obj, key)

        if (index % self.rate) != 0:
            return

        for key in self.dct.keys():
            if key in self.spike_vars:
                continue
            else:
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

        # increment spike count directly in dct
        for key in self.spike_vars:
            self.dct[key][:, d_index] += getattr(self.obj, key)

        if (index % self.rate) != 0:
            return

        for key in self.dct.keys():
            if key in self.spike_vars:
                continue
            else:
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
        self._update(index)

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

        for key in self.spike_vars:
            src = getattr(self.obj, key)
            self.gpu_dct[key][b_index] += src

        if index % self.rate != 0:
            return

        for key in self.dct.keys():
            src = getattr(self.obj, key)
            if key in self.spike_vars:
                continue
            else:
                dst = int(self.gpu_dct[key].gpudata) + b_index * src.nbytes
                cuda.memcpy_dtod(dst, src.gpudata, src.nbytes)

        # dump data to CPU if simulation complete or buffer full
        if (d_index == self.steps - 1) or (b_index == self.buffer_length - 1):
            for key in self.dct.keys():
                buffer = self.get_buffer(key, d_index)
                cuda.memcpy_dtoh(buffer, self.gpu_dct[key].gpudata)
            for key in self.spike_vars:
                self.gpu_dct[key].fill(0.0)

    def _copy_memory_dtoh(self, index):
        # downsample index
        d_index = int(index / self.rate)

        # increment spike count directly in dct
        for key in self.spike_vars:
            self.dct[key][:, d_index] += getattr(self.obj, key).get()

        if (index % self.rate) != 0:
            return

        for key in self.dct.keys():
            if key in self.spike_vars:
                continue
            else:
                self.dct[key][:, d_index] = getattr(self.obj, key).get()

    def _py2_get_buffer(self, key, index):
        beg = int(index / self.buffer_length) * self.buffer_length
        nbytes = self.gpu_dct[key].nbytes / self.buffer_length
        offset = int(beg * nbytes)
        size = int((index - beg + 1) * nbytes)
        return np.getbuffer(self.dct[key], offset, size)

    def _py3_get_buffer(self, key, index):
        mem_view = memoryview(self.dct[key].T)
        beg = int(index / self.buffer_length) * self.buffer_length
        return mem_view[beg : index + 1]
