#pylint:disable=abstract-method
"""Backend Modules for Model"""
import hashlib
import tempfile
import pathlib
import textwrap
from functools import cached_property
from abc import abstractmethod
import importlib.util
from types import MethodType
import numpy as np
import numba
import numba.cuda
from . import types as tpe
from . import errors as err
from .utils.array import cuda_fill, cuda_clip
from .codegen.numba import get_numba_function_source

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

class Backend:
    """Base Backend Implementation
    """
    def __init__(self, model: tpe.Model):
        self.model = model

class CUDABackend(Backend):

    blocksize: int = 256

    def __init__(
        self,
        model: tpe.Model
    ):
        if not numba.cuda.is_available():
            raise err.NeuralBackendError(
                "CUDA gpu not available, cannot support CUDABackend"
            )
        super().__init__(model)

    @property
    def gridsize(self) -> int:
        return int(min(
            6 * MULTIPROCESSOR_COUNT,
            (self.model.num-1) // self.blocksize + 1
        ))

    def recast(self) -> None:
        for attr in ["states", "gstates", "bounds", "params"]:
            for key, arr in (dct := getattr(self.model, attr)).items():
                if numba.cuda.is_cuda_array(arr):
                    dct[key] = numba.cuda.as_cuda_array(arr)
                else:
                    dct[key] = numba.cuda.to_device(np.asarray(arr))

    def reset(self) -> None:
        for attr in self.model.states:
            cuda_fill(self.model.states[attr], self.model.initial_states[attr])
        for attr in self.model.gstates:
            cuda_fill(self.model.gstates[attr], 0.)

    def clip(self, states: dict = None) -> None:
        states = self.model.states if states is None else states
        for var, bds in self.model.bounds.items():
            cuda_clip(states[var], *bds, states[var])

class CodegenBackend(Backend):
    """Backends with code generation into str

    For this backend, `generate()` populate :code:`self.source` with
    a :code:`str` definition of the compiled kernel functions.
    The functions are assumed to have the same same as the methods of
    :py:class:`neural.basemodel.Model`.

    If self.ode and/or self.post are defined, they are automatically populated
    """
    def __init__(self, model: tpe.Model):
        super().__init__(model)
        self.source = "" # source code for module
        self._module = None # ModuleType that is defined from the generated source code

    def compile(self):
        module_name = f"{self.__class__.__name__}For{self.model.__class__.__name__}"

        # determine which methods have yet to be implemented
        self.source += self.generate("ode")
        methods_to_implement = ["ode"]
        post = self.model.__class__.post
        for cls in self.model.__class__.__bases__:
            if cls.__name__ == "Model" and post != cls.post:
                methods_to_implement.append("post")
                self.source += self.generate("post")
                break

        # write source to temp file and load as module
        source = self.source.encode('utf-8')
        cache_dir = pathlib.Path(tempfile.gettempdir()) / \
            f"neural-compiler-cache-{_get_per_user_string()}"
        try:
            cache_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            pass
        except Exception as e:
            raise err.NeuralBackendError(
                f"Cannot creat cache dir {cache_dir}"
            ) from e
        checksum = hashlib.md5()
        checksum.update(source)
        cache_file = checksum.hexdigest()
        cache_path = cache_dir / f"{cache_file}.py"
        with open(cache_path, "w+b") as f:
            f.write(source)
            f.flush()
            try:
                spec = importlib.util.spec_from_file_location(module_name, cache_path)
                self._module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self._module)
            except Exception as e:
                raise err.NeuralBackendError(textwrap.dedent(
                    f"""Error in loading generated code for backend {self.__class__.__name__}
                    {self.source}
                    """
                )) from e

        # set globals for the module to match original function definition
        for method in methods_to_implement:
            for key, val in getattr(self.model, method).__globals__.items():
                setattr(self._module, key, val)

        # save method to backend referencing method to model
        for method in methods_to_implement:
            if not callable(func := getattr(self._module, method, None)):
                raise err.NeuralCodeGenError(
                    f"Method '{method}' not found in codegen module"
                )
            setattr(self, method, MethodType(func, self.model))

    @abstractmethod
    def generate(self, method: str) -> str:
        """Return generated str for a particular method"""

class NumbaCPUBackend(CodegenBackend):
    def generate(self, method: str) -> str:
        """generate source for numba kernel
        """
        if method not in ['ode', 'post']:
            raise err.NeuralCodeGenError(
                f"Only .ode and .post support codegen with numba, got '{method}'"
            )
        # jit target function
        try:
            return get_numba_function_source(self.model, method, target='numpy').src
        except Exception as e:
            raise err.NeuralCodeGenError(f"Code Generation for Method {method} failed") from e

class NumbaCUDABackend(CUDABackend, CodegenBackend):

    def __init__(
        self,
        model: tpe.Model
    ):
        if not numba.cuda.is_available():
            raise err.NeuralBackendError(
                "CUDA gpu not available, cannot support NumbaCUDABackend"
            )
        super().__init__(model)

    def generate(self, method: str) -> str:
        """generate source for numba kernel
        """
        if method not in ['ode', 'post']:
            raise err.NeuralCodeGenError(
                f"Only .ode and .post support codegen with numba, got '{method}'"
            )
        # jit target function
        try:
            return get_numba_function_source(self.model, method, target='cuda').src
        except Exception as e:
            raise err.NeuralCodeGenError(f"Code Generation for Method {method} failed") from e