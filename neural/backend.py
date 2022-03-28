# pylint:disable=abstract-method
"""Backend Modules for Model"""
import numpy as np
from functools import cache, update_wrapper
import hashlib
import tempfile
import pathlib
import sys
from abc import abstractmethod
import importlib.util
from types import MethodType, ModuleType
from warnings import warn
import numba
import numba.extending
import numba.cuda
from numba.cuda.dispatcher import Dispatcher
from . import errors as err
from .codegen.numba import get_numba_function_source

# copied from https://github.com/minrk/PyCUDA/blob/master/pycuda/compiler.py
def _get_per_user_string():
    try:
        from os import getuid
    except ImportError:
        checksum = hashlib.md5()
        from os import environ

        checksum.update(environ["HOME"])
        return checksum.hexdigest()
    else:
        return "uid%d" % getuid()


def _get_module_from_source(source: str, module_name: str) -> ModuleType:
    cache_dir = (
        pathlib.Path(tempfile.gettempdir())
        / f"neural-compiler-cache-{_get_per_user_string()}"
    )
    try:
        cache_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        pass
    except Exception as e:
        raise err.NeuralBackendError(f"Cannot creat cache dir {cache_dir}") from e

    checksum = hashlib.md5()
    checksum.update(source)
    cache_file = checksum.hexdigest()
    cache_path = cache_dir / f"{cache_file}.py"
    with open(cache_path, "w+b") as f:
        f.write(source)
        f.flush()
        try:
            spec = importlib.util.spec_from_file_location(module_name, cache_path)
            codegen_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(codegen_module)
            sys.modules[module_name] = codegen_module
        except Exception as e:
            raise err.NeuralBackendError(
                "Error in loading generated code for backend \n{}".format(
                    source.decode("utf-8")
                )
            ) from e
    return codegen_module


class BackendMixin:
    """Base Backend Implementation

    Properties:
        is_backend_supported

    Methods:
        clip() -> None
        reset() -> None
        recast() -> None
        compile() -> None
        _generate() -> None
    """

    @classmethod
    @property
    def is_backend_supported(cls) -> bool:
        return True


class CodegenBackendMixin(BackendMixin):
    """Backends with code generation into str

    For this backend, `_generate()` populate :code:`self.source` with
    a :code:`str` definition of the compiled kernel functions.
    The functions are assumed to have the same same as the methods of
    :py:class:`neural.basemodel.Model`.

    If self.ode and/or self.post are defined, they are automatically populated
    """

    codegen_source = ""  # source code for module
    codegen_module = None  # ModuleType that is defined from the generated source code

    def compile(self):
        self.codegen_source = ""
        self.codegen_module = None
        module_name = f"CodegenBackendMixin_For_{self.__class__.__name__}"

        # determine which methods have yet to be implemented
        self.codegen_source += self._generate("ode")
        methods_to_implement = ["ode"]

        # check if post is implemented by the child class
        post = self.__class__.post
        for cls in self.__class__.__bases__:
            if cls.__name__ == "Model" and post != cls.post:
                methods_to_implement.append("post")
                self.codegen_source += self._generate("post")
                break

        # write source to temp file and load as module
        source = self.codegen_source.encode("utf-8")
        self.codegen_module = _get_module_from_source(source, module_name)

        # set globals for the module to match original function definition
        for method in methods_to_implement:
            for key, val in getattr(self, method).__globals__.items():
                setattr(self.codegen_module, key, val)

        # save method to backend referencing method to model
        for method in methods_to_implement:
            if not callable(func := getattr(self.codegen_module, method, None)):
                raise err.NeuralCodeGenError(
                    f"Method '{method}' not found in codegen module"
                )
            setattr(self, method, MethodType(func, self))

    @abstractmethod
    def _generate(self, method: str) -> str:
        """Return generated str for a particular method"""


class NumbaCPUBackendMixin(CodegenBackendMixin):
    def _generate(self, method: str) -> str:
        """generate source for numba kernel"""
        if method not in ["ode", "post"]:
            raise err.NeuralCodeGenError(
                f"Only .ode and .post support codegen with numba, got '{method}'"
            )
        # jit target function
        try:
            return get_numba_function_source(self, method, target="numpy")
        except Exception as e:
            raise err.NeuralCodeGenError(
                f"Code Generation for Method {method} failed"
            ) from e

    def compile(self):
        super().compile()
        for method in ["ode", "post"]:
            func = getattr(self, method).__func__
            if numba.extending.is_jitted(func) or isinstance(func, Dispatcher):
                setattr(self, method, MethodType(func, self._data))


class NumbaCUDABackendMixin(CodegenBackendMixin):
    @classmethod
    @property
    def is_backend_supported(cls) -> bool:
        if not (supported := numba.cuda.is_available()):
            warn(
                "CUDA Backend requires CUDA-compatible GPU.",
                err.NeuralBackendWarning,
            )
        return supported

    @property
    def blocksize(self) -> int:
        return 256

    @cache
    def _get_gridsize(self, num: int) -> int:
        COUNT = numba.cuda.get_current_device().MULTIPROCESSOR_COUNT
        return int(min(6 * COUNT, (num - 1) // self.blocksize + 1))

    @property
    def gridsize(self) -> int:
        return self._get_gridsize(self.num)

    def recast(self) -> None:
        """
        FIXME:
            Due to limitation of str literal indexing of numba cuda array,
            and the fact that CuPY does not support structured arrays.
            The `states, gstates, params` variables will have to be
            return as host-side numpy arrays
        """
        if not numba.cuda.is_cuda_array(self._data):
            self._data = numba.cuda.to_device(self._data)
        # allocate pinned array for fast copy
        # self._data_cpu = numba.cuda.pinned_array_like(
        #     self._data.copy_to_host()
        # )

    @property
    def states(self) -> np.ndarray:
        state_vars = list(self.Default_States.keys())
        self._data.copy_to_host(to=self._data_cpu)
        return self._data_cpu[state_vars]

    @property
    def gstates(self) -> np.ndarray:
        gstate_vars = [f"d_{s}" for s in self.Derivates]
        self._data.copy_to_host(to=self._data_cpu)
        return self._data_cpu[gstate_vars]

    @property
    def params(self) -> np.ndarray:
        param_vars = list(self.Default_Params.keys())
        self._data.copy_to_host(to=self._data_cpu)
        return self._data_cpu[param_vars]

    def _generate(self, method: str) -> str:
        """generate source for numba kernel"""
        if method not in ["ode", "post"]:
            raise err.NeuralCodeGenError(
                f"Only .ode and .post support codegen with numba, got '{method}'"
            )
        # jit target function
        try:
            return get_numba_function_source(self, method, target="cuda")
        except Exception as e:
            raise err.NeuralCodeGenError(
                f"Code Generation for Method {method} failed"
            ) from e

    def compile(self):
        super().compile()

        # save method to backend referencing method to model
        for method in ["ode", "post"]:
            func = getattr(self, method).__func__
            if numba.extending.is_jitted(func) or isinstance(func, Dispatcher):
                func = func.configure(self.gridsize, self.blocksize)
                setattr(self, method, MethodType(func, self._data))
