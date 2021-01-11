# pylint: disable=no-member
import importlib
import jinja2
from .logger import NeuralBackendError

cuda_repeat_template = jinja2.Template(
    """

#define THREADS_PER_BLOCK 1024

__global__ void repeat(
    int num,
    int repeat,
    {{ dtype }} *input,
    {{ dtype }} *output
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;
    int start, end;

    int tot_num = num*repeat;
    __shared__ {{ dtype }} buffer[THREADS_PER_BLOCK/2];

    for(int i = tid; i < tot_num; i += total_threads)
    {
        // if (tid >= tot_num)
        // return;

        start = (i - threadIdx.x) / repeat;
        end = (i - threadIdx.x + blockDim.x - 1) / repeat;

        if (threadIdx.x <= (end - start))
            buffer[threadIdx.x] = input[start+threadIdx.x];
        __syncthreads();

        output[i] = buffer[i/repeat - start];
        __syncthreads();
    }
    return;
}
"""
)

class _PyCuda_SkCuda_merge:
    """Merged API from pycuda.cumath and skcuda.misc

    Note: THIS IS A HACK! It is meant to simplify (and attempt to speed up)
    the way `neural.math` finds the right math operator to use.
    """
    def __init__(self):
        self.cuda = importlib.import_module('pycuda.driver')
        self.garray = importlib.import_module('pycuda.gpuarray')
        self.cumath = importlib.import_module('pycuda.cumath')
        self.skmisc = importlib.import_module('skcuda.misc')
        self.skmisc.init()
        self.sklinalg = importlib.import_module('skcuda.linalg')
        self.cucompiler = importlib.import_module('pycuda.compiler')
        self._repeat = self.init_repeat()
        self._repeat_grid, self._repeat_block = None, None

    def __getattr__(self, method):
        try:
            return getattr(self.cumath, method)
        except AttributeError:
            pass
        
        try:
            return getattr(self.skmisc, method)
        except AttributeError:
            pass

        try:
            return getattr(self.sklinalg, method)
        except AttributeError:
            raise NeuralBackendError(f"Function {method} not found in pycuda.cumath/skcuda.misc/skcuda.linalg")

    def init_repeat(self):
        mod = self.cucompiler.SourceModule(
            cuda_repeat_template.render(dtype='double'),
            options=["--ptxas-options=-v"],
        )
        func = mod.get_function("repeat")
        func.prepare("iiPP")
        return func

    def repeat(self, arr, rep_size, size, output=None):
        if output is None:
            output_size = rep_size * size
            output = self.garray.empty((output_size,), dtype=arr.dtype)
        if self._repeat_grid is None:
            self._repeat_block = (1024,1,1)
            grid_x = 30 * self.cuda.Context.get_device().MULTIPROCESSOR_COUNT
            self._repeat_grid = (int(min(grid_x, (output_size - 1) / self._repeat_block[0] + 1)), 1)

        self._update.prepared_async_call(
            self._repeat_grid,
            self._repeat_block,
            None,
            size,
            rep_size,
            arr.gpudata,
            output.gpudata,
        )
        return output