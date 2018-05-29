import numpy as np

import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.curandom

from pycuda.compiler import SourceModule

from skcuda.fft import fft, Plan, ifft

src_cuda = """
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

extern "C"{


__global__ void  generate_seed(
    int num,
    curandState *seed
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = tid; i < num; i += total_threads)
        curand_init(clock64(), i, 0, &seed[i]);

    return;
}

__global__ void  generate_spike(
    int num,
    double dt,
    double rate,
    curandState *seed,
    double *scale,
    double *spike
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    float r;
    for (int i = tid; i < num; i += total_threads) {
        r = (curand_uniform(&seed[i]));

        spike[i] = double(r < dt*rate*scale[i]);
    }

    return;
}
}
"""

mod = SourceModule(src_cuda,
    options = ["--ptxas-options=-v"],
    no_extern_c = True
)
_generate_spike = mod.get_function("generate_spike")
_generate_spike.prepare('iddPPP')

_generate_seed = mod.get_function("generate_seed")
_generate_seed.prepare('iP')

class CUDASpikeGenerator(object):
    def __init__(self, dt, num, scale, **kwargs):
        self.dtype = kwargs.pop('dtype', np.float64)

        self.dt = dt
        self.num = num
        self.scale = scale
        self.scale = self.scale.astype(self.dtype)

        self.gpu_seed = pycuda.driver.mem_alloc(num * 48)
        self.gpu_scale = gpuarray.to_gpu(self.scale)
        self.spike = gpuarray.zeros(num, dtype=self.dtype)

        self.block = (128,1,1)
        self.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                    (self.num-1) / self.block[0] + 1), 1)

        self._generate_spike = _generate_spike
        self._generate_seed = _generate_seed
        self._generate_seed.prepared_async_call(
            self.grid, self.block, None, np.int32(self.num), self.gpu_seed)


    def generate(self, rate):
        self._generate_spike.prepared_async_call(
            self.grid, self.block, None, np.int32(self.num), self.dt, rate,
            self.gpu_seed, self.gpu_scale.gpudata, self.spike.gpudata)

def cu_lpf(stimulus, dt, freq):
    """
    CUDA implementation of low-pass-filter.

    stimulus: ndarray
        The input to be filtered.
    dt: float
        The sampling interval of the input.
    freq: float
        The cut-off frequency of the low pass filter.
    """
    num = len(stimulus)
    num_fft = num/2 + 1
    idtype = stimulus.dtype
    odtype = np.complex128 if idtype == np.float64 else np.complex64

    if not isinstance(stimulus, gpuarray.GPUArray):
        d_stimulus = gpuarray.to_gpu(stimulus)
    else:
        d_stimulus = stimulus

    plan = Plan(stimulus.shape, idtype, odtype)
    d_fstimulus = gpuarray.empty(num_fft, odtype)
    fft(d_stimulus, d_fstimulus, plan)

    df = 1./dt/num
    idx = int(freq//df)

    unit = d_fstimulus.dtype.itemsize / 4
    offset = int(d_fstimulus.gpudata) + d_fstimulus.dtype.itemsize * idx

    cuda.memset_d32(offset, 0, unit*(num_fft-idx))

    plan = Plan(stimulus.shape, odtype, idtype)
    d_lpf_stimulus = gpuarray.empty(num, idtype)
    ifft(d_fstimulus, d_lpf_stimulus, plan, True)

    return d_lpf_stimulus.get()
