"""
Utility modules for recording data from Neural models
"""
import numpy as np
import pycuda
import pycuda.gpuarray as garray
import pycuda.driver as cuda

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
    def __init__(self, model, attrs, steps, gpu_buffer=False):
        self.model = model
        self.steps = steps
        self.num = model.cuda_kernel.num

        if gpu_buffer:
            self.gpu_dct = {}
            for key in attrs:
                dtype = self.model.gdata[key].dtype
                self.gpu_dct[key] = garray.zeros((self.num, self.steps), dtype)
            self.copy_memory = self._copy_memory_dtod
            self.dct = {key: None for key in attrs}
        else:
            self.dct = {key: np.zeros((self.num, self.steps)) for key in attrs}
            self.copy_memory = self._copy_memory_dtoh

        self.block = (256, 1, 1)
        self.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                    (self.num-1) / self.block[0] + 1), 1)

    def __iter__(self):
        for i in xrange(self.steps):
            self.copy_memory(i)
            yield i

    def _copy_memory_dtod(self, index):
        for key in self.dct.keys():
            dtype = dtype_to_ctype(self.model.gdata[key].dtype)
            src = self.model.gdata[key].gpudata
            dst = int(self.gpu_dct[key].gpudata)
            _copy[dtype].prepared_async_call(self.grid, self.block, None,
                self.num, index, self.steps, dst, src)

        if index == self.steps-1:
            for key in self.dct.keys():
                self.dct[key] = self.gpu_dct[key].get()

    def _copy_memory_dtoh(self, index):
        for key in self.dct.keys():
            self.dct[key][:,index] = getattr(self.model, key).get()

    def __getitem__(self, key):
        return self.dct[key]

    def __getattr__(self, key):
        if key in self.dct:
            return self.dct[key]
        return super(CUDARecorder, self).__getattribute__(key)
