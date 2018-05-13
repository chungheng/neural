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
        self.dct = {key: np.zeros((self.num, self.steps)) for key in attrs}

        if gou_buffer:
            self.gpu_dct = garray.zeros((self.num, self.steps))

    def __iter__(self):
        for i in xrange(self.steps):
            for key in self.dct.keys():
                self.dct[key][:,i] = getattr(self.model, key).get()
                yield i

    def __getitem__(self, key):
        return self.dct[key]

    def __getattr__(self, key):
        if key in self.dct:
            return self.dct[key]
        return super(CUDARecorder, self).__getattribute__(key)
