"""
Utility modules for recording data from Neural models
"""
import sys
import time
import numpy as np
import pycuda
import pycuda.gpuarray as garray
import pycuda.driver as cuda


PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3


from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype

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

_copy = {'float': None, 'double': None}

for key, val in _copy.items():
    mod = SourceModule(src_cuda % {'type': key},
        options = ["--ptxas-options=-v"])
    func = mod.get_function('copy')
    func.prepare('iiiPP')
    _copy[key] = func

class CUDARecorder(object):
    """
    Recorder for reading CUDA arrays of Neural models.

    Attributes:
    """
    def __init__(self, obj, attrs, steps, **kwargs):
        self.obj = obj
        self.total_steps = steps
        self.rate = kwargs.pop('rate', 1)
        self.steps = int(steps/self.rate)
        gpu_buffer = kwargs.pop('gpu_buffer', False)
        callback = kwargs.pop('callback', False)

        self.dct = {}
        for key in attrs:
            src = getattr(self.obj, key)
            shape = (src.size, self.steps)
            self.dct[key] = np.zeros(shape, order='F', dtype=src.dtype)

        if gpu_buffer:
            self.buffer_length = self._get_buffer_length(gpu_buffer)
            self.gpu_dct = {}
            for key in attrs:
                src = getattr(self.obj, key)
                shape = (self.buffer_length, src.size)
                self.gpu_dct[key] = garray.zeros(shape, dtype=src.dtype)
            self.copy_memory = self._copy_memory_dtod
        else:
            self.copy_memory = self._copy_memory_dtoh

        if PY2:
            self.get_buffer = self._py2_get_buffer
        if PY3:
            self.get_buffer = self._py3_get_buffer

        if callback:
            self.iter = iter(self)
            func = lambda: next(self.iter)
            self.obj.cuda.callbacks.append(func)

    def reset(self):
        self.iter = iter(self)

    def __iter__(self):
        for i in range(self.total_steps):
            if i % self.rate == 0:
                self.copy_memory(i)
            yield i

    def _get_buffer_length(self, gpu_buffer):
        if gpu_buffer == 'full' or gpu_buffer == 'whole' or gpu_buffer is True:
            return self.steps
        else:
            return min(gpu_buffer, self.steps)

    def _copy_memory_dtod(self, index):
        # downsample index
        d_index = int(index/self.rate)
        # buffer index
        b_index = d_index % self.buffer_length
        for key in self.dct.keys():
            src = getattr(self.obj, key)
            dst = int(self.gpu_dct[key].gpudata) + b_index * src.nbytes

            cuda.memcpy_dtod(dst, src.gpudata, src.nbytes)

        if (d_index == self.steps-1) or (b_index == self.buffer_length-1):
            for key in self.dct.keys():
                buffer = self.get_buffer(key, d_index)
                cuda.memcpy_dtoh(buffer, self.gpu_dct[key].gpudata)

    def _copy_memory_dtoh(self, index):
        d_index = int(index/self.rate) # downsample index
        for key in self.dct.keys():
            self.dct[key][:,d_index] = getattr(self.obj, key).get()

    def __getitem__(self, key):
        return self.dct[key]

    def __getattr__(self, key):
        if key in self.dct:
            return self.dct[key]
        return super(CUDARecorder, self).__getattribute__(key)

    def _py2_get_buffer(self, key, index):
        beg = int(index / self.buffer_length) * self.buffer_length
        nbytes = self.gpu_dct[key].nbytes / self.buffer_length
        offset = int(beg * nbytes)
        size = int((index-beg+1) * nbytes)
        return np.getbuffer(self.dct[key], offset, size)

    def _py3_get_buffer(self, key, index):
        mv = memoryview(self.dct[key].T)
        beg = int(index / self.buffer_length) * self.buffer_length
        return mv[beg:index+1]
