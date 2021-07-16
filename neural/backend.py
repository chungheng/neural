# pylint: disable=no-member
"""
Backend Classes For Model

Backend Classes can define a variety of behaviors for the Models,
including (but not limited to):

1. Compilation
2. Resetting
3. Updating
4. Profiling
"""
from __future__ import print_function
from itertools import chain
from numbers import Number
import sys
from types import MethodType
from six import StringIO, get_function_globals
import numpy as np
from pycuda.elementwise import ElementwiseKernel

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

try:
    from .codegen.optimizer import FuncGenerator
except ImportError:
    FuncGenerator = None

try:
    from .codegen.optimizer import NumpyKernelGenerator
except ImportError:
    NumpyKernelGenerator = None

try:
    import pycuda
    import pycuda.driver as drv
    import pycuda.gpuarray as garray
    from pycuda.compiler import SourceModule
    from .codegen.cuda import CudaKernelGenerator
except ImportError:
    CudaKernelGenerator = None

try:
    import cupy as cp
except ImportError:
    CupyKernelGenerator = None

from .logger import NeuralBackendError

# copied from https://github.com/minrk/PyCUDA/blob/master/pycuda/compiler.py
def _new_md5():
    try:
        import hashlib

        return hashlib.md5()
    except ImportError:
        # for Python << 2.5
        import md5

        return md5.new()


# copied from https://github.com/minrk/PyCUDA/blob/master/pycuda/compiler.py
def _get_per_user_string():
    try:
        from os import getuid
    except ImportError:
        checksum = _new_md5()
        from os import environ

        checksum.update(environ["HOME"])
        return checksum.hexdigest()
    else:
        return "uid%d" % getuid()


class Backend(object):
    """Backend Base Class"""

    def __new__(cls, model, **kwargs):
        """
        Factory for instantiating different backends.
        """
        if cls is Backend:
            backend = kwargs.pop("backend", None)
            if backend == "scalar":
                if FuncGenerator is None:
                    raise  NeuralBackendError(
                        "PyCodegen is not installed."
                    )
                return super(Backend, cls).__new__(ScalarBackend)
            elif backend == "numpy":
                if NumpyKernelGenerator is None:
                    raise NeuralBackendError(
                        "PyCodegen is not installed."
                    )
                return super(Backend, cls).__new__(NumpyBackend)
            elif backend == "cuda":
                if CudaKernelGenerator is None:
                    raise  NeuralBackendError(
                        "Either PyCUDA or PyCodegen is not installed."
                    )
                return super(Backend, cls).__new__(CUDABackend)
            else:
                raise NeuralBackendError(f"Unexpected backend '{backend}'")
        return super(Backend, cls).__new__(cls)

    def compile(self):
        pass

    def clip(self, value, a_min, a_max) -> None:
        """Clip value in-place in [a_min,a_max] range"""


class ScalarBackend(Backend):
    def __init__(self, model, **kwargs):
        ostream = StringIO()
        code_gen = FuncGenerator(model, model.ode, offset=4, ostream=ostream)
        code_gen.generate()

        post = model.__class__.post
        for cls in model.__class__.__bases__:
            if cls.__name__ == "Model" and post != cls.post:
                code_gen = FuncGenerator(model, post, offset=4, ostream=ostream)
                code_gen.generate()
                break

        self.source = ostream.getvalue()
        self.func_globals = get_function_globals(model.ode)
        self.name = "Optimized{}".format(model.__class__.__name__)
        self.compile()

        self.ode = MethodType(self.module.ode, model)
        if "post" in self.module.__dict__:
            self.post = MethodType(self.module.post, model)

    def compile(self):

        from os import mkdir
        from os.path import join, isfile
        from tempfile import gettempdir

        cache_dir = join(
            gettempdir(), "pyneural-compiler-cache-%s" % _get_per_user_string()
        )

        try:
            mkdir(cache_dir)
        except OSError as e:
            from errno import EEXIST

            if e.errno != EEXIST:
                raise

        source = self.source.encode("utf-8")

        checksum = _new_md5()

        checksum.update(source)

        cache_file = checksum.hexdigest()
        cache_path = join(cache_dir, cache_file + ".py")

        if not isfile(cache_path):
            outf = open(cache_path, "wb")
            outf.write(source)
            outf.close()

        if PY2:
            import imp

            self.module = imp.load_source(self.name, cache_path)

        elif PY3:
            import importlib.util

            spec = importlib.util.spec_from_file_location(self.name, cache_path)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)

        for key, val in self.func_globals.items():
            setattr(self.module, key, val)

    def clip(self, value, a_min, a_max):
        np.clip(value, a_min, a_max, out=value)

class NumpyBackend(ScalarBackend):
    def __init__(self, model, **kwargs):
        self.num = kwargs.pop("num", None)

        ostream = StringIO()
        code_gen = NumpyKernelGenerator(model, model.ode, offset=4, ostream=ostream)
        code_gen.generate()

        post = model.__class__.post
        for cls in model.__class__.__bases__:
            if cls.__name__ == "Model" and post != cls.post:
                code_gen = NumpyKernelGenerator(model, post, offset=4, ostream=ostream)
                code_gen.generate()
                break

        self.source = ostream.getvalue()
        self.func_globals = get_function_globals(model.ode)
        self.name = "Numpy{}".format(model.__class__.__name__)
        self.compile()

        self.ode = MethodType(self.module.ode, model)
        if "post" in self.module.__dict__:
            self.post = MethodType(self.module.post, model)

    def reset(self, **kwargs):
        """
        reset the numpy data.

        Reset the Numpy data to default values.

        Arguments:
            kwargs (dict): keyward arguments.
        """
        params = []
        items = chain(self.model.params.items())
        for key, val in self.model.params.items():
            val = kwargs.pop(key, val)
            if hasattr(val, "__len__"):  # params with __len__
                if self.num != len(val):
                    raise NeuralBackendError(f"Instance has {self.num} units, but '{key}' has {len(val)} entires")
                if not isinstance(val, np.ndarray):
                    self.model.params[key] = np.asarray(val)

        for key, val in self.model.states.items():
            val = kwargs.pop(key, val)

            if hasattr(val, "__len__"):  # params with __len__
                if self.num != len(val):
                    raise NeuralBackendError(f"Instance has {self.num} units, but '{key}' has {len(val)} entires")
                self.model.states[key] = np.asarray(val, dtype=self.dtype)
            elif isinstance(val, Number):
                self.model.states[key] = np.ones(self.num, dtype=self.dtype) * val
            else:
                raise NeuralBackendError(f"Invalid '{key}' variable: {val}")

    def clip(self, value, a_min, a_max):
        np.clip(value, a_min, a_max, out=value)


class CUDABackend(Backend):
    def __init__(self, model, **kwargs):
        self.backend = kwargs.pop("backend", None)
        self.num = kwargs.pop("num", None)
        self.data = dict()
        self.model = model
        self.dtype = kwargs.pop("dtype", np.float64)
        self.compile(**kwargs)

        self._clip = ElementwiseKernel(
            "float *value, float a_min, float a_max",
            "value[i] = value[i] > a_min ? (value[i] < a_max ? value[i] : a_max) : a_min",
            "clip"
        )

    def _allocate_cuda_memory(self, key):
        """
        allocate GPU memroy for variable
        """
        if key in self.data and len(self.data[key]) != self.num:
            del self.data[key]

        if key not in self.data:
            array = garray.empty(self.num, dtype=self.dtype)
            self.data[key] = array

    def reset(self, **kwargs):
        """
        reset the gpu data.

        Reset the GPU data to default values.

        Arguments:
            kwargs (dict): keyward arguments.
        """
        params = []
        items = chain(self.model.states.items(), self.model.params.items())
        for key, val in items:
            val = kwargs.pop(key, val)

            # allocate GPU memory
            if key in self.model.states:
                self._allocate_cuda_memory(key)
            elif hasattr(val, "__len__"):  # params with __len__
                self._allocate_cuda_memory(key)
                params.append(key)

            if isinstance(val, np.ndarray):
                if val.dtype != self.dtype:
                    val = val.astype(self.dtype)
                drv.memcpy_htod(
                    self.data[key].gpudata, val
                )
            elif isinstance(val, garray.GPUArray):
                if key in self.model.params:
                    if val.dtype != self.dtype:
                        raise NeuralBackendError(
                            f"Model Param '{key}' has dtype {val.dtype} but Backend expects {self.dtype}"
                        )
                    self.data[key] = val
                    continue
                if val.dtype != self.dtype:
                    val = val.get()
                    val = val.astype(self.dtype)
                    drv.memcpy_htod(
                        self.data[key].gpudata, val
                    )
                else:
                    drv.memcpy_dtod(
                        self.data[key].gpudata, val.gpudata, val.nbytes
                    )
            elif isinstance(val, Number):
                if key in self.model.params:
                    self.model.params[key] = val
                    continue
                self.data[key].fill(val)
            else:
                raise NeuralBackendError(f"Invalid variable '{key}': {val}")
        return params

    def compile(self, **kwargs):
        """
        compile the cuda kernel.

        Keyword Arguments:
        """

        # decide the number of threads:
        keys = chain(self.model.states.keys(), self.model.params.keys())
        for key in keys:
            val = getattr(self.model, key)
            val = kwargs.get(key, val)
            if hasattr(val, "__len__"):
                _num = len(val)
                self.num = self.num or _num
                if self.num != _num:
                    raise NeuralBackendError(f"Mismatch in data size: {key}")
        else:
            if not self.num:
                raise NeuralBackendError("Please give the number of models to run")

        # reset gpu data
        params = self.reset(**kwargs)

        # assume the rest of kwargs are input-related
        inputs = kwargs.copy()
        for key in inputs.keys():
            if key not in self.model.Inputs:
                NeuralBackendError(f"Unexpected input '{key}'")

        # generate cuda kernel, a.k.a self.cuda.kernel
        self.get_cuda_kernel(inputs_gdata=inputs, params_gdata=params)

        if self.has_random:
            self.seed = drv.mem_alloc(self.num * 48)
            self.init_random_seed.prepared_async_call(
                self.grid, self.block, None, self.num, self.seed
            )

    def update(self, d_t, **kwargs):
        """Update State Variables"""
        st = kwargs.pop("st", None)
        args = []
        for key, dtype in zip(self.args, self.arg_ctype[2:]):
            if key == "seed":
                args.append(self.seed)
                continue

            val = self.data.get(key, None)
            try:
                if val is None:
                    val = kwargs[key]
                if hasattr(val, "__len__"):
                    if dtype != "P":
                        raise NeuralBackendError(f"[{self.model}] Expect GPU array but get a scalar input: {key}")
                    if val.dtype != self.dtype:
                        raise NeuralBackendError(f"[{self.model}] GPU array float type mismatches: {key}")
                else:
                    if dtype == "P":
                        raise NeuralBackendError(f"[{self.model}] Expect GPU array but get a scalar input: {key}")
            except Exception as e:
                raise NeuralBackendError(f"{self.model} Error") from e

            args.append(val.gpudata if dtype == "P" else val)

        self.kernel.prepared_async_call(self.grid, self.block, st, self.num, d_t, *args)

    def cuda_profile(self, **kwargs):
        num = kwargs.pop("num", 1000)
        niter = kwargs.pop("niter", 1000)
        dtype = kwargs.pop("dtype", np.float64)

        self.cuda_compile(num=num, dtype=dtype)

        args = {key: garray.empty(num, dtype) for key in self.cuda.args}

        start = drv.Event()
        end = drv.Event()
        secs = 0.0

        for i in range(niter):
            start.record()
            self._cuda_update(0.0, **args)
            end.record()
            end.synchronize()
            secs += start.time_till(end)

        for key in args:
            args[key].gpudata.free()

        name = self.__class__.__name__
        print("Average run time of {}: {} ms".format(name, secs / niter))

    def get_cuda_kernel(self, **kwargs):
        code_generator = CudaKernelGenerator(self.model, dtype=self.dtype, **kwargs)
        code_generator.generate()

        try:
            mod = SourceModule(
                code_generator.cuda_src,
                options=["--ptxas-options=-v", "--expt-relaxed-constexpr"],
                no_extern_c=code_generator.has_random,
            )
            func = mod.get_function(self.model.__class__.__name__)
        except:
            lines = code_generator.cuda_src.split("\n")
            num_digits = 1 + int(np.floor(np.log10(len(lines))))
            for index, line in enumerate(lines):
                print("{: >{}}: {}".format(index, num_digits, line))
            raise NeuralBackendError("CUDA Kernel Generation Error")

        self.src = code_generator.cuda_src
        self.args = code_generator.args
        self.arg_ctype = code_generator.arg_type

        func.prepare(self.arg_ctype)
        self.kernel = func

        self.has_random = code_generator.has_random
        if self.has_random:
            init_random_seed = mod.get_function("generate_seed")
            init_random_seed.prepare(code_generator.init_random_seed_arg)
            self.init_random_seed = init_random_seed

        deviceData = pycuda.tools.DeviceData()
        maxThreads = int(np.float(deviceData.registers // func.num_regs))
        maxThreads = int(2 ** int(np.log(maxThreads) / np.log(2)))
        threadsPerBlock = int(min(256, maxThreads, deviceData.max_threads))
        self.block = (threadsPerBlock, 1, 1)
        self.grid = (
            int(
                min(
                    6
                    * drv.Context.get_device().MULTIPROCESSOR_COUNT,  # pylint: disable=no-member
                    (self.num - 1) / threadsPerBlock + 1,
                )
            ),
            1,
        )

        return func


    def clip(self, value, a_min, a_max):
        if isinstance(value, garray.GPUArray):
            self._clip(value, a_min, a_max)
        else:
            np.clip(value, a_min, a_max, out=value)
