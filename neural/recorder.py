"""
Utility modules for recording data from Neural models
"""
import numpy as np

class CUDARecorder(object):
    """
    Recorder for reading CUDA arrays of Neural models.
    """
    def __init__(self, model, attrs, steps):
        self.model = model
        self.steps = steps
        self.num = model.cuda_kernel.num
        self.dct = {key: np.zeros((self.num, self.steps)) for key in attrs}

    def __iter__(self):
        for i in xrange(self.steps):
            yield i
            for key in self.dct.keys():
                self.dct[key][:,i] = getattr(self.model, key).get()

    def __getitem__(self, key):
        return self.dct[key]

    def __getattr__(self, key):
        if key in self.dct:
            return self.dct[key]
        return super(CUDARecorder, self).__getattribute__(key)
