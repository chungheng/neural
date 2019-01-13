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

if PY2:
    getbuffer = np.getbuffer
if PY3:
    def getbuffer(obj, offset, size):
        mv = memoryview(obj)
        return mv[offset:offset+size]

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
    """
    def __init__(self, model, attrs, steps, **kwargs):
        self.model = model
        self.steps = steps
        self.num = model.cuda_kernel.num
        gpu_buffer = kwargs.pop('gpu_buffer', False)
        callback = kwargs.pop('callback', False)

        if gpu_buffer:
            self.buffer_length = self._get_buffer_length(gpu_buffer)
            self.gpu_dct = {}
            self.dct = {}
            for key in attrs:
                dtype = self.model.gdata[key].dtype
                self.gpu_dct[key] = garray.zeros((self.buffer_length,
                    self.num), dtype)
                self.dct[key] = np.zeros((self.num, self.steps),
                    order='F', dtype=dtype)
            self.copy_memory = self._copy_memory_dtod
        else:
            self.dct = {key: np.zeros((self.num, self.steps)) for key in attrs}
            self.copy_memory = self._copy_memory_dtoh

        self.block = (256, 1, 1)
        self.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                    (self.num-1) / self.block[0] + 1), 1)

        if callback:
            self.iter = iter(self)
            func = lambda: next(self.iter)
            self.model.cuda_kernel.callbacks.append(func)

    def __iter__(self):
        for i in range(self.steps):
            self.copy_memory(i)
            yield i

    def _get_buffer_length(self, gpu_buffer):
        if gpu_buffer == 'full' or gpu_buffer == 'whole' or gpu_buffer is True:
            return self.steps
        else:
            return min(gpu_buffer, self.steps)

    def _copy_memory_dtod(self, index):
        for key in self.dct.keys():
            dtype = dtype_to_ctype(self.model.gdata[key].dtype)
            src = self.model.gdata[key].gpudata
            nbytes = self.model.gdata[key].nbytes
            dst = int(self.gpu_dct[key].gpudata)

            idx = index % self.buffer_length
            cuda.memcpy_dtod(dst + idx * nbytes, src, nbytes)

        end = index + 1

        if index == self.steps-1 or (end % self.buffer_length == 0):

            for key in self.dct.keys():
                beg = (index / self.buffer_length) * self.buffer_length
                nbytes = self.gpu_dct[key].nbytes / self.buffer_length
                offset = int(beg * nbytes)
                size = int((end-beg) * nbytes)
                buffer = getbuffer(self.dct[key], offset, size)
                cuda.memcpy_dtoh(buffer, self.gpu_dct[key].gpudata)

    def _copy_memory_dtoh(self, index):
        for key in self.dct.keys():
            self.dct[key][:,index] = getattr(self.model, key).get()

    def __getitem__(self, key):
        return self.dct[key]

    def __getattr__(self, key):
        if key in self.dct:
            return self.dct[key]
        return super(CUDARecorder, self).__getattribute__(key)
