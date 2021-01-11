# pylint: disable=no-member
"""
Utility modules for recording data from Neural models
"""
from abc import abstractmethod
from numbers import Number
import sys
import typing as tp
import numpy as np

import pycuda.gpuarray as garray
import pycuda.driver as cuda
from .logger import NeuralRecorderError
from . import config

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3


class Recorder(object):
    """
    Base recorder module.

    Attributes:
    """

    def __new__(cls, obj, attrs: tp.Iterable[str], steps: int, **kwargs):
        if cls is Recorder:
            attr = getattr(obj, attrs[0])
            if config.BACKEND == 'scalar':
                return super(Recorder, cls).__new__(ScalarRecorder)
            if config.BACKEND == 'numpy':
                return super(Recorder, cls).__new__(NumpyRecorder)
            if config.BACKEND == 'pycuda':
                return super(Recorder, cls).__new__(CUDARecorder)
            if config.BACKEND == 'cupy':
                return super(Recorder, cls).__new__(CuPyRecorder)
            raise NeuralRecorderError(
                f"""{attr} of type: {type(attr)} not understood. 
                Only Number, Numpy NDArray and PyCuda GpuArrays are accepted"""
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
        self.steps = int(steps / self.rate)

        self.dct = {key: None for key in attrs}

        # handle spike
        self._spike_recorder = dict()
        for atr in attrs:
            if "spike" in atr.lower():
                self._spike_recorder[atr] = 0
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

    @abstractmethod
    def update(self, index: int):
        """
        update the record
        """

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
            if not isinstance(src, Number):
                raise NeuralRecorderError(
                    f"ScalarRecorder got src={src} for key={key} of obj={self.obj}, needs to be a Number."
                )
            self.dct[key] = np.zeros(self.steps, dtype=type(src))

    def update(self, index: int):
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
            if not isinstance(src, np.ndarray):
                raise NeuralRecorderError(
                    f"NumpyRecorder got src={src} for key={key} of obj={self.obj}, needs to be a Numpy NDArray."
                )
            shape = (src.size, self.steps)
            self.dct[key] = np.zeros(shape, order="F", dtype=src.dtype)

    def update(self, index: int):
        d_index = int(index / self.rate)  # downsample index
        for key in self.dct.keys():
            self.dct[key][:, d_index] = getattr(self.obj, key)


class CUDARecorder(Recorder):
    """
    Recorder for reading CUDA arrays of Neural models.

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
        super(CUDARecorder, self).__init__(
            obj, attrs, steps, rate=rate, callback=callback
        )

        if self._spike_recorder:
            for key in self._spike_recorder:
                src = getattr(self.obj, key)
                self._spike_recorder[key] = garray.zeros(src.shape, order="C", dtype=src.dtype)

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
            if not isinstance(src, garray.GPUArray):
                raise NeuralRecorderError(
                    f"CUDARecorder got src={src} for key={key} of obj={self.obj}, needs to be a PyCuda GPUArray."
                )
            shape = (src.size, self.steps)
            self.dct[key] = cuda.pagelocked_zeros(shape, order="C", dtype=src.dtype)
            # self.dct[key] = np.zeros(shape, order="C", dtype=src.dtype)

    def update(self, index: int):
        # update spike counter if has spike attributes
        if self._spike_recorder:
            for key in self._spike_recorder:
                self._spike_recorder[key] += getattr(self.obj, key)

        if (index % self.rate) != 0:
            return
        
        # if should record, go ahead
        self._update(index)

    def _get_buffer_length(self, gpu_buffer):
        if gpu_buffer == "full" or gpu_buffer == "whole" or gpu_buffer is True:
            return self.steps
        else:
            return min(gpu_buffer, self.steps)

    def _copy_memory_dtod(self, index):
        # downsample index
        d_index = int(index // self.rate)
        # buffer index
        b_index = d_index % self.buffer_length
        for key in self.dct.keys():
            if key in self._spike_recorder:
                src = self._spike_recorder[key]
            else:
                src = getattr(self.obj, key)
            dst = int(self.gpu_dct[key].gpudata) + b_index * src.nbytes
            cuda.memcpy_dtod(dst, src.gpudata, src.nbytes)

        # move to host if recording complete or buffer full
        if (d_index == self.steps - 1) or (b_index == self.buffer_length - 1):
            for key in self.dct.keys():
                # if key in self._spike_recorder:
                #     self._spike_recorder[key].get(ary=self.dct[key][:, d_index])
                # else:
                buffer = self.get_buffer(key, d_index)
                try:
                    cuda.memcpy_dtoh(buffer, self.gpu_dct[key].gpudata) 
                except Exception as e:
                    import pdb; pdb.set_trace()

    def _copy_memory_dtoh(self, index):
        # downsample index
        d_index = int(index // self.rate)
        for key in self.dct.keys():
            if key in self._spike_recorder:
                self._spike_recorder[key].get(ary=self.dct[key][:, d_index])
            else:
                getattr(self.obj, key).get(ary=self.dct[key][:, d_index])

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
        super().__init__(
            obj, attrs, steps, rate=rate, callback=callback
        )

        if self._spike_recorder:
            for key, val in self._spike_recorder.items():
                src = getattr(self.obj, key)
                self._spike_recorder[key] = garray.zeros_like(src)

        if gpu_buffer:
            self.buffer_length = self._get_buffer_length(gpu_buffer)
            self.gpu_dct = {}
            for key in attrs:
                src = getattr(self.obj, key)
                shape = (self.buffer_length, src.size)
                if "spike" in key:  # spike, spike_state
                    continue
                else:
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
            if not isinstance(src, garray.GPUArray):
                raise NeuralRecorderError(
                    f"CUDARecorder got src={src} for key={key} of obj={self.obj}, needs to be a PyCuda GPUArray."
                )
            shape = (src.size, self.steps)
            self.dct[key] = cuda.pagelocked_zeros(shape, order="F", dtype=src.dtype)

    def update(self, index: int):
        # update spike counter if has spike attributes
        if self._spike_recorder:
            for key in self._spike_recorder:
                self._spike_recorder[key] += getattr(self.obj, key)

        if (index % self.rate) != 0:
            return
        self._update(index)
        # for key in self._spike_recorder:
        #     self._spike_recorder[key].fill(0.)

    def _get_buffer_length(self, gpu_buffer):
        if gpu_buffer == "full" or gpu_buffer == "whole" or gpu_buffer is True:
            return self.steps
        else:
            return min(gpu_buffer, self.steps)

    def _copy_memory_dtod(self, index):
        # downsample index
        d_index = int(index // self.rate)
        # buffer index
        b_index = d_index % self.buffer_length
        for key in self.dct.keys():
            if key in self._spike_recorder:
                continue
            src = getattr(self.obj, key)
            dst = int(self.gpu_dct[key].gpudata) + b_index * src.nbytes
            cuda.memcpy_dtod(dst, src.gpudata, src.nbytes)

        # move to host if recording complete or buffer full
        if (d_index == self.steps - 1) or (b_index == self.buffer_length - 1):
            for key in self.dct.keys():
                if key in self._spike_recorder:
                    self._spike_recorder[key].get(ary=self.dct[key][:, d_index])
                else:
                    buffer = self.get_buffer(key, d_index)
                    cuda.memcpy_dtoh(buffer, self.gpu_dct[key].gpudata)

    def _copy_memory_dtoh(self, index):
        # downsample index
        d_index = int(index // self.rate)
        for key in self.dct.keys():
            if key in self._spike_recorder:
                self._spike_recorder[key].get(ary=self.dct[key][:, d_index])
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
