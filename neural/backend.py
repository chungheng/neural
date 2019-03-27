"""
Base model class for neurons and synapses.
"""
from __future__ import print_function
from abc import abstractmethod
from collections import OrderedDict
from numbers import Number
import sys

from six import StringIO, with_metaclass
import numpy as np

PY2 = sys.version_info[0] == 3
PY3 = sys.version_info[0] == 3

if PY2:
    from inspect import getargspec as _getfullargspec
    varkw = 'keywords'
if PY3:
    from inspect import getfullargspec as _getfullargspec
    varkw = 'varkw'

try:
    from .codegen.optimizer import FuncGenerator
except ImportError:
    FuncGenerator = None

try:
    import pycuda
    import pycuda.gpuarray as garray
    from pycuda.tools import dtype_to_ctype
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    from .codegen.cuda import CudaKernelGenerator, get_func_signature
except ImportError:
    CudaKernelGenerator = None
    pycuda = None


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
    def __init__(self):
        pass

    def compile(self):
        pass

    def update(self):
        pass

class CUDABackend(object):
    def __init__(self, model, dtype=np.float64, num=None, **kwargs):
        self.num = num
        self.data = dict()
        self.model = model
        self.dtype = dtype
        self.compile(**kwargs)

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
        for key, attr in self.model.Variables.items():
            dct = getattr(self.model, attr)
            val = kwargs.pop(key, dct[key])

            # allocate GPU memory
            if attr != 'params':
                self._allocate_cuda_memory(key)
            elif hasattr(val, '__len__'): # params with __len__
                self._allocate_cuda_memory(key)
                params.append(key)

            if isinstance(val, np.ndarray):
                if val.dtype != self.dtype:
                    val = val.astype(self.dtype)
                drv.memcpy_htod(self.data[key].gpudata, val)
            elif isinstance(val, garray.GPUArray):
                if attr == 'params':
                    assert val.dtype == self.dtype
                    self.data[key] = val
                    continue
                if val.dtype != self.dtype:
                    val = val.get()
                    val = val.astype(self.dtype)
                    drv.memcpy_htod(self.data[key].gpudata, val)
                else:
                    drv.memcpy_dtod(self.data[key].gpudata, val.gpudata)
            elif isinstance(val, Number):
                if attr == 'params':
                    self.model.params[key] = val
                    continue
                self.data[key].fill(val)
            else:
                raise TypeError("Invalid {0} variable: {1}".format(attr, key))
        return params

    def compile(self, **kwargs):
        """
        compile the cuda kernel.

        Keyword Arguments:
        """

        # decide the number of threads:
        for key in self.model.Variables.keys():
            val = getattr(self.model, key)
            val = kwargs.get(key, val)
            if hasattr(val, '__len__'):
                _num = len(val)
                self.num = self.num or _num
                assert self.num == _num, 'Mismatch in data size: %s' % key
        else:
            assert self.num, 'Please give the number of models to run'

        # reset gpu data
        params = self.reset(**kwargs)

        # assume the rest of kwargs are input-related
        inputs = kwargs.copy()
        for key in inputs.keys():
            assert key in self.model.Inputs, "Unexpected input '{}'".format(key)

        # generate cuda kernel, a.k.a self.cuda.kernel
        self.get_cuda_kernel(inputs_gdata=inputs, params_gdata=params)

        if self.has_random:
            self.seed = drv.mem_alloc(self.num * 48)
            self.init_random_seed.prepared_async_call(
                self.grid,
                self.block,
                None,
                self.num,
                self.seed)

    def update(self, d_t, **kwargs):
        """

        """
        st = kwargs.pop('st', None)
        args = []
        for key, dtype in zip(self.args, self.arg_ctype[2:]):
            if key == 'seed':
                args.append(self.seed)
                continue

            val = self.data.get(key, None)
            if val is None:
                val = kwargs[key]
            if hasattr(val, '__len__'):
                assert dtype == 'P', \
                    "Expect GPU array but get a scalar input: %s" % key
                assert val.dtype == self.dtype, \
                    "GPU array float type mismatches: %s" % key
            else:
                assert dtype != 'P', \
                    "Expect GPU array but get a scalar input: %s" % key

            args.append(val.gpudata if dtype == 'P' else val)

        self.kernel.prepared_async_call(
            self.grid,
            self.block,
            st,
            self.num,
            d_t,
            *args)

    def cuda_profile(self, **kwargs):
        num = kwargs.pop('num', 1000)
        niter = kwargs.pop('niter', 1000)
        dtype = kwargs.pop('dtype', np.float64)

        self.cuda_compile(num=num, dtype=dtype)

        args = {key: garray.empty(num, dtype) for key in self.cuda.args}

        start = drv.Event()
        end = drv.Event()
        secs = 0.

        for i in range(niter):
            start.record()
            self._cuda_update(0., **args)
            end.record()
            end.synchronize()
            secs += start.time_till(end)

        for key in args:
            args[key].gpudata.free()

        name = self.__class__.__name__
        print('Average run time of {}: {} ms'.format(name, secs/niter))

    def get_cuda_kernel(self, **kwargs):
        assert CudaKernelGenerator is not None

        code_generator = CudaKernelGenerator(self.model,
            dtype=self.dtype, **kwargs)
        code_generator.generate()

        try:
            mod = SourceModule(code_generator.cuda_src,
                options = ["--ptxas-options=-v"],
                no_extern_c = code_generator.has_random)
            func = mod.get_function(self.model.__class__.__name__)
        except:
            lines = code_generator.cuda_src.split('\n')
            num_digits = 1 + int(np.floor(np.log10(len(lines))))
            for index, line in enumerate(lines):
                print("{: >{}}: {}".format(index, num_digits, line))
            raise

        self.src = code_generator.cuda_src
        self.args = code_generator.args
        self.arg_ctype = code_generator.arg_type

        func.prepare(self.arg_ctype)
        self.kernel = func

        self.has_random = code_generator.has_random
        if self.has_random:
            init_random_seed = mod.get_function('generate_seed')
            init_random_seed.prepare(code_generator.init_random_seed_arg)
            self.init_random_seed = init_random_seed

        deviceData = pycuda.tools.DeviceData()
        maxThreads = int(np.float(deviceData.registers // func.num_regs))
        maxThreads = int(2**int(np.log(maxThreads) / np.log(2)))
        threadsPerBlock = int(min(256, maxThreads, deviceData.max_threads))
        self.block = (threadsPerBlock, 1, 1)
        self.grid = (
            int(min(6 * drv.Context.get_device().MULTIPROCESSOR_COUNT,
                (self.num-1) / threadsPerBlock + 1)),
            1)

        return func
