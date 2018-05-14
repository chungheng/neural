"""
Utility modules for recording data from Neural models
"""
import numpy as np
import pycuda
import pycuda.gpuarray as garray
import pycuda.driver as cuda

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
                self.gpu_dct[key] = garray.zeros((self.steps, self.num), dtype)
            self.copy_memory = self._copy_memory_dtod
            self.dct = {key: None for key in attrs}
        else:
            self.dct = {key: np.zeros((self.num, self.steps)) for key in attrs}
            self.copy_memory = self._copy_memory_dtoh

    def __iter__(self):
        for i in xrange(self.steps):
            self.copy_memory(i)
            yield i

    def _copy_memory_dtod(self, index):
        for key in self.dct.keys():
            size = self.model.gdata[key].nbytes
            src = self.model.gdata[key].gpudata
            dest = int(self.gpu_dct[key].gpudata) + size*index
            # print src, dest
            cuda.memcpy_dtod(dest, src, size)
        if index == self.steps:
            for key in self.dct.keys():
                self.dct[key] = self.gpu_dct[key].get().T
            # self.dct[key][:,index] = getattr(self.model, key).get()

    def _copy_memory_dtoh(self, index):
        for key in self.dct.keys():
            self.dct[key][:,index] = getattr(self.model, key).get()

    def __getitem__(self, key):
        return self.dct[key]

    def __getattr__(self, key):
        if key in self.dct:
            return self.dct[key]
        return super(CUDARecorder, self).__getattribute__(key)
