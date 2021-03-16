# pylint:disable=no-member
"""CUDA Utilities
"""
import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.curandom
from pycuda.compiler import SourceModule
from skcuda.fft import fft, Plan, ifft

src_random_cuda = """
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
src_repeat_cuda = """
#define THREADS_PER_BLOCK 1024

__global__ void repeat_float(
    int num,
    int repeat,
    float *input,
    float *output
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tot_num = num*repeat;
    __shared__ float buffer[THREADS_PER_BLOCK/2];

    if (tid >= tot_num)
        return;

    int start = blockIdx.x * blockDim.x / repeat;
    int end = int(ceil(float((blockIdx.x+1) * blockDim.x - 1) / repeat));

    if (threadIdx.x < (end - start))
        buffer[threadIdx.x] = input[start+threadIdx.x];
    __syncthreads();

    output[tid] = buffer[tid/repeat - start];
    return;
}

__global__ void repeat_double(
    int num,
    int repeat,
    double *input,
    double *output
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int tot_num = num*repeat;
    __shared__ double buffer[THREADS_PER_BLOCK/2];

    if (tid >= tot_num)
        return;

    int start = blockIdx.x * blockDim.x / repeat;
    int end = int(ceil(float((blockIdx.x+1) * blockDim.x - 1) / repeat));

    if (threadIdx.x < (end - start))
        buffer[threadIdx.x] = input[start+threadIdx.x];
    __syncthreads();

    output[tid] = buffer[tid/repeat - start];
    return;
}
"""

mod = SourceModule(src_random_cuda, options=["--ptxas-options=-v"], no_extern_c=True)
_generate_spike = mod.get_function("generate_spike")
_generate_spike.prepare("iddPPP")

_generate_seed = mod.get_function("generate_seed")
_generate_seed.prepare("iP")

mod = SourceModule(src_repeat_cuda, options=["--ptxas-options=-v"])

_repeat_float = mod.get_function("repeat_float")
_repeat_float.prepare("iiPP")

_repeat_double = mod.get_function("repeat_double")
_repeat_double.prepare("iiPP")


class CUDASpikeGenerator(object):
    """
    CUDA implementation of low-pass-filter.

    stimulus: ndarray
        The input to be filtered.
    dt: float
        The sampling interval of the input.
    freq: float
        The cut-off frequency of the low pass filter.
    """

    def __init__(self, dt, num, scale, **kwargs):
        self.dtype = kwargs.pop("dtype", np.float64)

        self.dt = dt
        self.num = num
        self.scale = scale
        self.scale = self.scale.astype(self.dtype)

        self.gpu_seed = pycuda.driver.mem_alloc(num * 48)
        self.gpu_scale = gpuarray.to_gpu(self.scale)
        self.spike = gpuarray.zeros(num, dtype=self.dtype)

        self.block = (128, 1, 1)
        self.grid = (
            min(
                6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                (self.num - 1) / self.block[0] + 1,
            ),
            1,
        )

        self._generate_spike = _generate_spike
        self._generate_seed = _generate_seed
        self._generate_seed.prepared_async_call(
            self.grid, self.block, None, np.int32(self.num), self.gpu_seed
        )

    def generate(self, rate):
        self._generate_spike.prepared_async_call(
            self.grid,
            self.block,
            None,
            np.int32(self.num),
            self.dt,
            rate,
            self.gpu_seed,
            self.gpu_scale.gpudata,
            self.spike.gpudata,
        )


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
    num_fft = int(num / 2 + 1)
    idtype = stimulus.dtype
    odtype = np.complex128 if idtype == np.float64 else np.complex64

    if not isinstance(stimulus, gpuarray.GPUArray):
        d_stimulus = gpuarray.to_gpu(stimulus)
    else:
        d_stimulus = stimulus

    plan = Plan(stimulus.shape, idtype, odtype)
    d_fstimulus = gpuarray.empty(num_fft, odtype)
    fft(d_stimulus, d_fstimulus, plan)

    df = 1.0 / dt / num
    idx = int(freq // df)

    unit = int(d_fstimulus.dtype.itemsize / 4)
    offset = int(d_fstimulus.gpudata) + d_fstimulus.dtype.itemsize * idx

    cuda.memset_d32(offset, 0, unit * (num_fft - idx))

    plan = Plan(stimulus.shape, odtype, idtype)
    d_lpf_stimulus = gpuarray.empty(num, idtype)
    ifft(d_fstimulus, d_lpf_stimulus, plan, False)

    return d_lpf_stimulus.get()


def cu_repeat(input, repeat, output=None):
    """
    Repeat a PyCUDA 1-D array.

    Parameters
    ----------
    input: pycuda.GPUArray
        the input to be repeated.
    repeat: int
        the number of repeatition. Only supports 1-D array now.
    output: pycuda.GPUArray, optional
        the output. If not given, the output will be created with length of
        ``len(input)*repeat''.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> a = gpuarray.to_gpu([1., 2. ,3.])
    >>> a_repeat = cu_repeat(a, 5)
    >>> numpy.allclose(numpy.repeat(a.get(), 5), a_repeat.get())
    """
    num = len(input)
    tot_num = num * repeat
    if output is None:
        output = gpuarray.empty(num * repeat, input.dtype)

    block = (1024, 1, 1)
    grid = (
        min(
            6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
            (tot_num - 1) / block[0] + 1,
        ),
        1,
    )

    if input.dtype == np.float32:
        _repeat_float.prepared_async_call(
            grid, block, None, num, repeat, input.gpudata, output.gpudata
        )
    elif input.dtype == np.float64:
        _repeat_double.prepared_async_call(
            grid, block, None, num, repeat, input.gpudata, output.gpudata
        )
    else:
        raise TypeError(repr(input.dtype))

    return output
