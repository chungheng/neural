# pylint:disable=abstract-method
"""Backend Modules for Model"""
from functools import update_wrapper
import hashlib
import tempfile
import pathlib
import textwrap
from abc import abstractmethod
import importlib.util
from types import MethodType
import numpy as np
import numba
import numba.extending
import numba.cuda
from . import errors as err
from .utils.array import cuda_fill, cuda_clip
from .codegen.numba import get_numba_function_source

try:
    import cupy as cp
except ImportError:
    cp = None

if numba.cuda.is_available():
    MULTIPROCESSOR_COUNT = numba.cuda.get_current_device().MULTIPROCESSOR_COUNT
else:
    MULTIPROCESSOR_COUNT = None

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


class BackendMixin:
    """Base Backend Implementation"""
    @classmethod
    @property
    def is_backend_supported(cls) -> bool:
        return True

class CuPyBackendMixin(BackendMixin):

    @classmethod
    @property
    def is_backend_supported(cls) -> bool:
        try:
            import cupy as cp
            return True
        except ImportError:
            return False


    def recast(self) -> None:
        for attr in ["states", "gstates", "bounds", "params"]:
            for key, arr in (dct := getattr(self, attr)).items():
                dct[key] = cp.asarray(arr)

    def reset(self) -> None:
        for attr in self.states:
            self.states[attr].fill(self.initial_states[attr])
        for attr in self.gstates:
            self.states[attr].fill(0.0)

    def clip(self, states: dict = None) -> None:
        states = self.states if states is None else states
        for var, bds in self.bounds.items():
            cp.clip(states[var], *bds, out=states[var])


class CodegenBackendMixin(BackendMixin):
    """Backends with code generation into str

    For this backend, `generate()` populate :code:`self.source` with
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
        self.codegen_source += self.generate("ode")
        methods_to_implement = ["ode"]
        post = self.__class__.post
        for cls in self.__class__.__bases__:
            if cls.__name__ == "Model" and post != cls.post:
                methods_to_implement.append("post")
                self.codegen_source += self.generate("post")
                break

        # write source to temp file and load as module
        source = self.codegen_source.encode("utf-8")
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
                self.codgen_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self.codgen_module)
            except Exception as e:
                raise err.NeuralBackendError(
                    f"Error in loading generated code for backend {self.__class__.__name__}"
                ) from e

        # set globals for the module to match original function definition
        for method in methods_to_implement:
            for key, val in getattr(self, method).__globals__.items():
                setattr(self.codgen_module, key, val)

        # save method to backend referencing method to model
        for method in methods_to_implement:
            if not callable(func := getattr(self.codgen_module, method, None)):
                raise err.NeuralCodeGenError(
                    f"Method '{method}' not found in codegen module"
                )
            setattr(self, method, MethodType(func, self))

    @abstractmethod
    def generate(self, method: str) -> str:
        """Return generated str for a particular method"""


class NumbaCPUBackendMixin(CodegenBackendMixin):
    def generate(self, method: str) -> str:
        """generate source for numba kernel"""
        if method not in ["ode", "post"]:
            raise err.NeuralCodeGenError(
                f"Only .ode and .post support codegen with numba, got '{method}'"
            )
        # jit target function
        try:
            return get_numba_function_source(self, method, target="numpy").src
        except Exception as e:
            raise err.NeuralCodeGenError(
                f"Code Generation for Method {method} failed"
            ) from e


class NumbaCUDABackendMixin(CodegenBackendMixin):
    @classmethod
    @property
    def is_backend_supported(cls) -> bool:
        return numba.cuda.is_available()

    @property
    def blocksize(self) -> int:
        return 256

    @property
    def gridsize(self) -> int:
        return int(
            min(6 * MULTIPROCESSOR_COUNT, (self.num - 1) // self.blocksize + 1)
        )

    def reset(self) -> None:
        for attr in self.states:
            cuda_fill[self.gridsize, self.blocksize](
                self.states[attr], self.initial_states[attr]
            )
        for attr in self.gstates:
            cuda_fill[self.gridsize, self.blocksize](
                self.states[attr], 0.0
            )

    def clip(self, states: dict = None) -> None:
        states = self.states if states is None else states
        for var, bds in self.bounds.items():
            cuda_clip[self.gridsize, self.blocksize](states[var], *bds, out=states[var])

    def generate(self, method: str) -> str:
        """generate source for numba kernel"""
        if method not in ["ode", "post"]:
            raise err.NeuralCodeGenError(
                f"Only .ode and .post support codegen with numba, got '{method}'"
            )
        # jit target function
        try:
            return get_numba_function_source(self, method, target="cuda").src
        except Exception as e:
            raise err.NeuralCodeGenError(
                f"Code Generation for Method {method} failed"
            ) from e

    def compile(self):
        super().compile()

        # save method to backend referencing method to model
        for method in ['ode', 'post']:
            func = getattr(self, method)
            if numba.extending.is_jitted(func):
                def wrapper(*args, **kwargs):
                    return func[self.gridsize, self.blocksize](*args, **kwargs)
                update_wrapper(wrapper, func)
                setattr(self, method, wrapper)