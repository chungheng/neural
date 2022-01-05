import jinja2
import numpy as np

try:
    import pycuda
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    from pycuda.elementwise import ElementwiseKernel
    import pycuda.cumath as cumath

    import pycuda.gpuarray as garray
    import skcuda
    import skcuda.misc
    import skcuda.linalg

    CUDA = True
except:
    CUDA = False

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


class Operator(object):
    def __init__(self, size=None, output_size=None, dtype=np.float64, backend="cuda"):
        self.size = size
        if output_size is None:
            output_size = size
        self._output_size = output_size
        self.dtype = "double" if dtype == np.float64 else "float"
        if self._output_size is not None:
            self.output = garray.empty(self._output_size, dtype=dtype)
        else:
            self.output = 0.0
        self._backend = backend
        if backend == "scalar":
            self.output = self.output.get()
        elif backend != "cuda":
            raise NotImplementedError("{} backend not understood".format(backend))

    def update(self, **kwargs):
        pass

    def compile(self, **kwargs):
        pass


class Add(Operator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, **input):
        args = input.values()
        self._update(self.output, *args)

    def compile(self, **kwargs):
        num = len(kwargs)
        ins = ["in{}".format(i) for i in range(num)]
        if self._backend == "cuda":
            self._update = ElementwiseKernel(
                ", ".join("{} *{}".format(self.dtype, x) for x in ["out"] + ins),
                "out[i] = {};".format(" + ".join(["{}[i]".format(x) for x in ins])),
                "aggregate",
            )
        else:
            raise NotImplementedError


class Square(Operator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, input):
        self._update(self.output, input)

    def compile(self, **kwargs):
        if self._backend == "cuda":
            self._update = ElementwiseKernel(
                "{dtype} *out, {dtype} *in".format(dtype=self.dtype),
                "out[i] = in[i]*in[i];",
                "square",
            )
        else:
            self._update = lambda x: x ** 2


class Sqrt(Operator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, input):
        self._update(self.output, input)

    def compile(self, **kwargs):
        if self._backend == "cuda":
            self._update = ElementwiseKernel(
                "{dtype} *out, {dtype} *in".format(dtype=self.dtype),
                "out[i] = sqrt(in[i]);",
                "Sqrt",
            )
        else:
            self._update = lambda x: np.sqrt(x)


class Sum(Operator):
    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.output = 0.0

    def update(self, input):
        if self._backend == "cuda":
            self.output = skcuda.misc.sum(input)
        else:
            self.output = np.sum(input)


class BlockSum(Operator):
    def __init__(self, **kwargs):
        block_size = kwargs.pop("block_size", 1)
        self.block_size = block_size
        kwargs["output_size"] = int(kwargs["size"] // block_size)
        super().__init__(**kwargs)

    def update(self, input):
        if np.isscalar(input):
            input = np.array([input])
        _input = input.reshape(-1, self.block_size)
        if self._backend == "cuda":
            skcuda.misc.sum(_input, out=self.output, axis=1)
        else:
            self.output = np.sum(_input, axis=1)


class Mean(Operator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.output = garray.zeros(1, dtype=dtype)

    def update(self, input):
        if self._backend == "cuda":
            self.output = skcuda.misc.mean(input)
        else:
            self.output = np.mean(input)


class BlockMean(Operator):
    def __init__(self, block_size=1, **kwargs):
        self.block_size = block_size
        kwargs["output_size"] = int(kwargs["size"] // block_size)
        super().__init__(**kwargs)

    def update(self, input):
        _input = input.reshape(-1, self.block_size)
        if self._backend == "cuda":
            skcuda.misc.mean(_input, out=self.output, axis=1)
        else:
            self.output = np.mean(_input, axis=1)


class Repeat(Operator):
    def __init__(self, rep_size, **kwargs):
        kwargs["output_size"] = rep_size * kwargs["size"]
        super().__init__(**kwargs)
        self.rep_size = rep_size

        if self._backend == "cuda":
            self.block = (1024, 1, 1)
            grid_x = 30 * drv.Context.get_device().MULTIPROCESSOR_COUNT
            self.grid = (
                int(min(grid_x, (self._output_size - 1) / self.block[0] + 1)),
                1,
            )
            mod = SourceModule(
                cuda_repeat_template.render(dtype=self.dtype),
                options=["--ptxas-options=-v"],
            )
            self._update = mod.get_function("repeat")
            self._update.prepare("iiPP")

    def update(self, input):
        if self._backend == "cuda":
            self._update.prepared_async_call(
                self.grid,
                self.block,
                None,
                self.size,
                self.rep_size,
                input.gpudata,
                self.output.gpudata,
            )
        else:
            self.output = np.repeat(input, self.rep_size)


class Dot(Operator):
    def __init__(self, multiplier=None, batch_size=1, **kwargs):
        kwargs["output_size"] = multiplier.shape[0] * batch_size
        super().__init__(**kwargs)
        if isinstance(multiplier, np.ndarray):
            multiplier = multiplier.astype(self.dtype)
            multiplier = np.asfortranarray(multiplier)
            if self._backend == "cuda":
                self.multiplier = garray.to_gpu(multiplier)
            else:
                self.multiplier = multiplier
        elif isinstance(multiplier, garray.GPUArray):
            self.multiplier = multiplier
        else:
            raise TypeError("Unexpected type of multiplier.")

        self.batch_size = batch_size
        self._output = self.output.reshape(-1, batch_size, order="F")

    def update(self, input):
        _input = input.reshape(-1, self.batch_size, order="F")
        if self._backend == "cuda":
            skcuda.linalg.dot(self.multiplier, _input, out=self._output)
        else:
            self._output = np.dot(self.multiplier, _input)
