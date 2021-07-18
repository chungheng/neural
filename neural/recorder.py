# pylint:disable=no-member
"""
Utility modules for recording data from Neural models
"""
from abc import abstractmethod
from numbers import Number
import sys
import typing as tp
import numpy as np
import cupy as cp

import pycuda.gpuarray as garray
import pycuda.driver as cuda
from .logger import NeuralRecorderError
from . import config
from . import types as tpe


class Recorder(object):
    """
    Base recorder module.

    Attributes:
    """

    def __new__(cls, obj, attrs: tp.Iterable[str], steps: int, **kwargs):
        if cls is Recorder:
            attr = getattr(obj, attrs[0])
            if isinstance(attr, Number):
                return super(Recorder, cls).__new__(ScalarRecorder)
            elif isinstance(attr, np.ndarray):
                return super(Recorder, cls).__new__(NumpyRecorder)
            elif isinstance(attr, garray.GPUArray):
                return super(Recorder, cls).__new__(CUDARecorder)
            elif isinstance(attr, cp.ndarray):
                return super(Recorder, cls).__new__(CuPyRecorder)
            raise NeuralRecorderError(
                f"{attr} of type: {type(attr)} not understood. "
                "Only Number, Numpy NDArray and PyCuda GpuArrays are accepted"
            )
        return super(Recorder, cls).__new__(cls)

    def __init__(
        self,
        obj,
        attrs: tp.Iterable[str],
        steps: int,
        rate: int = 1,
        callback: tp.Union[tp.Callable, tp.Iterable[tp.Callable]] = False,
        **kwargs,
    ):
        self.obj = obj
        self.total_steps = steps
        self.rate = rate
        self.steps = len(np.arange(steps)[:: self.rate])

        self.dct = {key: None for key in attrs}
        self.spike_vars = tuple([key for key in attrs if "spike" in key.lower()])
        self.init_dct()

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

    def update(self, index: int):
        """Update the content of the recorder dictionaries

        .. note::

            Storing binary variables like spike counts should
            accumulate across time bins. As such, all
            spike variables are updated every time step regardless
            of the :code:`rate` attribute. All other attributes
            are updated once every `rate` steps.

        """
        d_index = int(index / self.rate)  # downsample index

        # increment spike count directly in dct
        for key in self.spike_vars:
            self.dct[key][..., d_index] += getattr(self.obj, key)

        if (index % self.rate) != 0:
            return

        for key in self.dct.keys():
            if key in self.spike_vars:
                continue
            else:
                self.dct[key][..., d_index] = getattr(self.obj, key)

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
            if not isinstance(src, Number):
                raise NeuralRecorderError(
                    f"ScalarRecorder got src={src} for key={key} of obj={self.obj}, "
                    "needs to be a Number."
                )
            self.dct[key] = np.zeros(self.steps, dtype=type(src))


class NumpyRecorder(Recorder):
    """
    Recorder for reading Numpy arrays of Neural models.

    Attributes:
    """

    def init_dct(self):
        for key in self.dct.keys():
            src = getattr(self.obj, key)
            if not isinstance(src, np.ndarray):
                raise NeuralRecorderError(
                    f"NumpyRecorder got src={src} for key={key} of obj={self.obj}, "
                    "needs to be a Numpy NDArray."
                )
            shape = (src.size, self.steps)
            self.dct[key] = np.zeros(shape, order="F", dtype=src.dtype)


class CUDARecorder(Recorder):
    """
    Recorder for reading CUDA arrays of Neural models.

    Arguments:
        - obj
        - attrs
        - steps:
        - rate
        - callback
        - gpu_buffer
    """

    def __init__(
        self,
        obj,
        attrs,
        steps,
        rate: int = 1,
        callback=False,
        gpu_buffer: tp.Union[str, bool, int] = False,
    ):
        super().__init__(obj, attrs, steps, rate=rate, callback=callback)

        # initialize gpu_dct
        self.gpu_dct = {}
        if gpu_buffer:
            self.buffer_length = self._get_buffer_length(gpu_buffer)
            for key in attrs:
                src = getattr(self.obj, key)
                shape = (self.buffer_length, src.size)
                self.gpu_dct[key] = garray.zeros(shape, dtype=src.dtype)
            self._update = self._copy_memory_dtod
        else:
            self._update = self._copy_memory_dtoh
        self.get_buffer = self._py3_get_buffer

    def init_dct(self):
        for key in self.dct.keys():
            src = getattr(self.obj, key)
            if not isinstance(src, garray.GPUArray):
                raise NeuralRecorderError(
                    f"CUDARecorder got src={src} for key={key} of obj={self.obj}, "
                    "needs to be a PyCuda GPUArray."
                )
            shape = (src.size, self.steps)
            self.dct[key] = cuda.pagelocked_zeros(shape, order="F", dtype=src.dtype)

    def update(self, index: int):
        d_index = int(index / self.rate)  # downsample index

        # increment spike count directly in dct
        for key in self.spike_vars:
            if self.gpu_dct:
                self.gpu_dct[key][..., d_index] += getattr(self.obj, key)
            else:
                self.dct[key][..., d_index] += getattr(self.obj, key).get()

        if (index % self.rate) != 0:
            return

        self._update(index)

    def _get_buffer_length(self, gpu_buffer):
        if gpu_buffer in ["full", "whole", True]:
            return self.steps
        else:
            return min(gpu_buffer, self.steps)

    def _copy_memory_dtod(self, index):
        # downsample index
        d_index = int(index // self.rate)
        # buffer index
        b_index = d_index % self.buffer_length
        for key in self.dct.keys():
            if key in self.spike_vars:
                continue
            else:
                src = getattr(self.obj, key)
            dst = int(self.gpu_dct[key].gpudata) + b_index * src.nbytes
            cuda.memcpy_dtod(dst, src.gpudata, src.nbytes)

        # move to host if recording complete or buffer full
        if (d_index == self.steps - 1) or (b_index == self.buffer_length - 1):
            for key in self.dct.keys():
                buffer = self.get_buffer(key, d_index)
                cuda.memcpy_dtoh(buffer, self.gpu_dct[key].gpudata)

    def _copy_memory_dtoh(self, index):
        # downsample index
        d_index = int(index // self.rate)
        for key in self.dct.keys():
            getattr(self.obj, key).get(ary=self.dct[key][:, d_index])

    def _py3_get_buffer(self, key, index):
        mem_view = memoryview(self.dct[key].T)
        beg = int(index / self.buffer_length) * self.buffer_length
        return mem_view[beg : index + 1]


class CuPyRecorder(Recorder):
    """
    Recorder for reading CuPy arrays of Neural models.

    Attributes:
    """

    def __init__(
        self,
        obj,
        attrs,
        steps,
        rate: int = 1,
        callback=False,
        gpu_buffer: tp.Union[str, bool, int] = False,
    ):
        super().__init__(obj, attrs, steps, rate=rate, callback=callback)

        self.gpu_dct = {}
        if gpu_buffer:
            self.buffer_length = self._get_buffer_length(gpu_buffer)
            for key in attrs:
                src = getattr(self.obj, key)
                shape = (self.buffer_length, src.size)
                self.gpu_dct[key] = cp.zeros(shape, dtype=src.dtype)
        self.get_buffer = self._py3_get_buffer

    def init_dct(self):
        for key in self.dct.keys():
            src = getattr(self.obj, key)
            if not isinstance(src, cp.ndarray):
                raise NeuralRecorderError(
                    f"CuPyRecorder got src={src} for key={key} of obj={self.obj}, "
                    "needs to be a CuPy NdArray."
                )
            shape = (src.size, self.steps)
            self.dct[key] = np.zeros(shape, order="F", dtype=src.dtype)

    def update(self, index: int):
        d_index = int(index / self.rate)  # downsample index

        # increment spike count directly in dct
        for key in self.spike_vars:
            if self.gpu_dct:
                self.gpu_dct[key][..., d_index] += getattr(self.obj, key)
            else:
                self.dct[key][..., d_index] += getattr(self.obj, key).get()

        if (index % self.rate) != 0:
            return

        if self.gpu_dct:
            for key in self.gpu_dct:
                if key in self.spike_vars:
                    continue
                self.gpu_dct[..., d_index] = getattr(self.obj, key)
            if (d_index == self.steps - 1) or (b_index == self.buffer_length - 1):
                for key in self.dct.keys():
                    buffer = self.get_buffer(key, d_index)
                    self.gpu_dct[key].get(out=buffer)
        else:
            self.dct[..., d_index] = getattr(self.obj, key).get()

    def _get_buffer_length(self, gpu_buffer):
        if gpu_buffer in ["full", "whole", True]:
            return self.steps
        else:
            return min(gpu_buffer, self.steps)

    def _py3_get_buffer(self, key, index):
        mem_view = memoryview(self.dct[key].T)
        beg = int(index / self.buffer_length) * self.buffer_length
        return mem_view[beg : index + 1]
