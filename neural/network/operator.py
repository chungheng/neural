import jinja2
import numpy as np
import pycuda
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import pycuda.cumath as cumath
import pycuda.gpuarray as garray
import skcuda
import skcuda.misc
import skcuda.linalg

cuda_repeat_template = jinja2.Template("""

#define THREADS_PER_BLOCK 1024

__global__ void repeat(
    int num,
    int repeat,
    {{ dtype }} *input,
    {{ dtype }} *output
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int tot_num = num*repeat;
    __shared__ {{ dtype }} buffer[THREADS_PER_BLOCK/2];

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
""")

class Add(object):
    def __init__(self, size, dtype=np.float64):
        self.output = garray.empty(size, dtype=dtype)
        self.dtype = 'double' if dtype == np.float64 else 'float'

    def update(self, **input):
        args = input.values()
        self._update(self.output, *args)

    def compile(self, **kwargs):
        num = len(kwargs)
        ins = ["in{}".format(i) for i in range(num)]
        self._update = ElementwiseKernel(
            ", ".join("{} *{}".format(self.dtype, x) for x in ['out'] + ins),
            "out[i] = {};".format(" + ".join(["{}[i]".format(x) for x in ins])),
            "aggregate")

class Square(object):
    def __init__(self, size, dtype=np.float64):
        self.output = garray.empty(size, dtype=dtype)
        self.dtype = 'double' if dtype == np.float64 else 'float'

    def update(self, input):
        self._update(self.output, input)

    def compile(self, **kwargs):
        self._update = ElementwiseKernel(
            "{dtype} *out, {dtype} *in".format(dtype=self.dtype),
            "out[i] = in[i]*in[i];",
            "square")

class Sqrt(object):
    def __init__(self, size, dtype=np.float64):
        self.output = garray.empty(size, dtype=dtype)
        self.dtype = 'double' if dtype == np.float64 else 'float'

    def update(self, input):
        self._update(self.output, input)

    def compile(self, **kwargs):
        self._update = ElementwiseKernel(
            "{dtype} *out, {dtype} *in".format(dtype=self.dtype),
            "out[i] = sqrt(in[i]);",
            "Sqrt")


class Sum(object):
    def __init__(self, dtype=np.float64):
        pass
    def update(self, input):
        self.output = skcuda.misc.sum(input)
        # print(skcuda.misc.sum(input), skcuda.misc.sum(input).dtype)

class BlockSum(object):
    def __init__(self, size, block_size=1, dtype=np.float64):
        self.block_size = block_size
        self.size = size
        self.output = garray.empty(int(size//block_size), dtype=dtype)
    def update(self, input):
        _input = input.reshape(-1, self.block_size)
        skcuda.misc.sum(_input, out=self.output, axis=1)

class Mean(object):
    def __init__(self, dtype=np.float64):
        self.output = garray.empty(1, dtype=dtype)
    def update(self, input):
        skcuda.misc.sum(input, out=self.output)

class BlockMean(object):
    def __init__(self, size=1, block_size=1, dtype=np.float64):
        self.block_size = block_size
        self.size = size
        self.output = garray.empty(int(size//block_size), dtype=dtype)
    def update(self, input):
        _input = input.reshape(-1, self.block_size)
        skcuda.misc.sum(_input, out=self.output, axis=1)

class Repeat(object):
    def __init__(self, size, rep_size, dtype=np.float64):
        dtype = 'double' if dtype == np.float64 else 'float'
        output_size = rep_size*size
        self.size = size
        self.rep_size = rep_size
        self.output = garray.empty(int(output_size), dtype=dtype)

        self.block = (1024,1,1)
        self.grid = (int(min(6 * drv.Context.get_device().MULTIPROCESSOR_COUNT,
            (output_size-1) / self.block[0] + 1)), 1)

        mod = SourceModule(
            cuda_repeat_template.render(dtype=dtype),
            options = ["--ptxas-options=-v"]
        )

        self._update = mod.get_function("repeat")
        self._update.prepare('iiPP')

    def update(self, input):
        self._update.prepared_async_call(
            self.grid, self.block, None,
            self.size, self.rep_size, input.gpudata, self.output.gpudata
        )

class Multiply(object):
    def __init__(self, multiplier, dtype=np.float64):
        if isinstance(multiplier, np.ndarray):
            multiplier = multiplier.astype(dtype)
            self.multiplier = garray.to_gpu(multiplier)
        elif isinstance(multiplier, garray.GPUArray):
            self.multiplier = multiplier
        else:
            raise TypeError("Unexpected type of multiplier.")

        self.output = garray.empty(multiplier.shape[0], dtype=dtype)
        self._output = self.output.reshape(-1, 1)

    def update(self, input):
        _input = input.reshape(-1, 1)
        skcuda.linalg.dot(self.multiplier, input, out=self._output)
