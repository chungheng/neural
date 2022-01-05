# pylint: disable=no-member
# pylint: disable=import-outside-toplevel
# pylint: disable=global-statement
"""Neural Global Configuration

The top-level configuration module of Neural defines variables that
configures the package:

1. CUDA (bool): Indicates if a CUDA compatible GPU is being used
2. INITIALIZED (bool): Indicates if the backend has been initialized
3. BACKEND (str): numpy/pycuda/cupy that is set when `init` is called

Two Functions are provided at global scope:

1. :code:`init(backend="numpy") -> None`: Initialize Neural's backend as specified.
    Calling this function will also set the 3 global variables
2. :code:`cuda_available() -> bool`: Returns a boolean flag of whether a cuda compatible
    gpu is found. The check relies on either PyCuda or CuPy being installed

Two Decorator is provided:

1. :code:`neural_initialzed`: use this decorator to enforce :code:`INITIALIZED=True`
    when a function is being called. Note that not all functions would require the
    backend be initialized.
2. :code:`with_backend(backend)`: use this decorator to enforce a specific backend
    when a function is called. Will raise :code:`NeuralBackendError` if the backend
    specified globally is not the same as the one in the decorator argument.
"""
from warnings import warn
import importlib
import functools
from .basemodel import Model
from .errors import NeuralBackendError, NeuralBackendWarning
from ._pycuda_skcuda import _PyCuda_SkCuda_merge

CUDA = False
INITIALIZED = False
BACKEND = None
_pycuda_skcuda_merge = None
_math_engine = None


def init(backend: str = "numpy") -> None:
    """
    Initialize Neural and Backend

    Keyword Arguments:
        backend: a str that configures the backend of the entire neural runtime context
            - numpy: no GPU is used
            - pycuda: use pycuda for GPU support
            - cupy: use cupy for GPU support

    Re-run with different backend str to re-configure backend
    """
    global CUDA
    global INITIALIZED
    global BACKEND
    global _math_engine
    global _pycuda_skcuda_merge
    # Reset to default in-case a re-init is called
    CUDA = False
    INITIALIZED = False
    BACKEND = None
    _math_engine = None
    if backend == "scalar":
        import math

        CUDA = False
        INITIALIZED = True
        BACKEND = backend
        _math_engine = math

    elif backend == "numpy":
        try:
            import numpy
        except ImportError as e:
            raise NeuralBackendError(
                "NumPy specified as backend but not installed"
            ) from e
        CUDA = False
        INITIALIZED = True
        BACKEND = backend
        _math_engine = numpy
        return
    elif backend == "pycuda":
        try:
            import pycuda
        except ImportError as e:
            raise NeuralBackendError(
                "PyCUDA specified as backend but not installed"
            ) from e

        try:
            import skcuda.misc

            skcuda.misc.init()
        except ImportError as e:
            raise NeuralBackendError(
                "PyCUDA specified as backend, Scikit-CUDA is required but not installed"
            ) from e
        except Exception as e:
            raise NeuralBackendError("Scikit-CUDA Initialization Failed") from e

        try:
            import pycuda.autoinit
        except pycuda.driver.RuntimeError as e:
            raise NeuralBackendError("PyCUDA AutoInit Failed") from e
        except Exception as e:
            raise NeuralBackendError("PyCUDA AutoInit Failed with unknown error") from e

        # use a hack to go through both pycuda.cumath and skcuda.misc
        print("initialization pycuda skcuda backend", flush=True)
        if _pycuda_skcuda_merge is None:
            _pycuda_skcuda_merge = _PyCuda_SkCuda_merge()
        print("done initialization", flush=True)
        CUDA = True
        INITIALIZED = True
        BACKEND = backend
        _math_engine = _pycuda_skcuda_merge
        return
    elif backend == "cupy":
        try:
            import cupy
        except ImportError as e:
            raise NeuralBackendError(
                "CuPy specified as backend but not installed"
            ) from e
        except Exception as e:
            raise NeuralBackendError("CuPy Import Failed") from e

        try:
            cupy.cuda.runtime.getDeviceCount()
        except cupy.cuda.runtime.CUDARuntimeError as e:
            warn(
                NeuralBackendWarning(
                    f"CuPy initialization failed, forcing CPU mode. {e}"
                )
            )
            CUDA = False
            INITIALIZED = True
            BACKEND = backend
            _math_engine = cupy
        except Exception as e:
            raise NeuralBackendError("CuPy Runtime Check failed") from e
        else:
            CUDA = True
            INITIALIZED = True
            BACKEND = backend
            _math_engine = cupy
        return
    else:
        raise NeuralBackendError(
            f"Backend {backend} Not understood, only scalar/numpy/pycuda/cupy are supported."
        )


def cuda_available() -> bool:
    """Check if CUDA is available"""
    try:
        import pycuda
    except ImportError:
        pass
    else:
        try:
            import pycuda.autoinit

            return True
        except pycuda.driver.RuntimeError:
            return False
    try:
        import cupy
    except ImportError:
        return False
    else:
        try:
            cupy.cuda.runtime.getDeviceCount()
            return True
        except cupy.cuda.runtime.CUDARuntimeError:
            return False


def neural_initialized(func):
    """Decorator to enforce neural initialization"""

    @functools.wraps(func)
    def function(*args, **kwargs):
        if not INITIALIZED:
            raise NeuralBackendError(
                "Neural Not Initialized, call neural.init(backend='') first."
            )
        return func(*args, **kwargs)

    return function


def with_backend(backend: str):
    """Decorator to enforce a specific backend"""

    def actual_wrapper(func):
        @functools.wraps(func)
        def function(*args, **kwargs):
            if not BACKEND != backend:
                raise NeuralBackendError(
                    "Neural Not Initialized, call neural.init(backend='') first."
                )
            return func(*args, **kwargs)

        return function

    return actual_wrapper


@neural_initialized
def backend_array_module():
    if BACKEND == "numpy":
        return importlib.import_module("numpy")
    if BACKEND == "cupy":
        return importlib.import_module("cupy")
    if BACKEND == "pycuda":
        return importlib.import_module("pycuda.gpuarray")
    return None
