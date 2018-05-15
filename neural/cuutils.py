import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.curandom
from pycuda.compiler import SourceModule

g_spk_generate_cuda = """
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

extern "C"{

__global__ void  spike_generate(
    int num,
    double dt,
    double rate,
    curandState_t *seed,
    double *scale,
    double *spike
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    double r;

    curandState_t state;

    for (int i = tid; i < num; i += total_threads) {
        state = seed[i];

        r = curand_uniform(&state);

        spike[i] = double(r < dt*rate*scale[i]);
        seed[i] = state;
    }

    return;
}
}
"""

mod = SourceModule(g_spk_generate_cuda,
    options = ["--ptxas-options=-v"],
    no_extern_c = True
)
_spk_generate = mod.get_function("spike_generate")
_spk_generate.prepare('iDDPPP')

class CUDASpikeGenerator(object):
    def __init__(self, dt, num, scale, **kwargs):
        self.dtype = kwargs.pop('dtype', np.float64)

        self.dt = dt
        self.num = num
        self.scale = scale
        self.scale.astype(self.dtype)

        self.gpu_seed = pycuda.curandom.seed_getter_unique(self.num)
        self.gpu_scale = gpuarray.to_gpu(self.scale)
        self.spike = gpuarray.empty_like(self.gpu_scale)

        self.block = (256,1,1)
        self.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                    (self.num-1) / 256 + 1), 1)
                    
        self._spk_generate = mod.get_function("spike_generate")



    def generate(self, rate):
        self._spk_generate.prepared_async_call(
            self.grid, self.block, None, self.num, self.dt, self.dtype(rate),
            self.gpu_seed.gpudata, self.gpu_scale.gpudata, self.spike.gpudata)
        return self.spike
