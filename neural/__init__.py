from warnings import warn
from .logger import NeuralBackendError, NeuralBackendWarning

CUDA = False
INITIALIZED = False


def init(backend: str = "numpy") -> None:
    """
    Initialize Neural and Backend

    Keyword Arguments:
        backend: a str that configures the backend of the entire neural runtime context
            - numpy: no GPU is used
            - pycuda: use pycuda for GPU support
            - cupy: use cupy for GPU support
    """
    global CUDA
    global INITIALIZED
    if backend == "numpy":
        CUDA = False
        INITIALIZED = True
        return

    if backend == "pycuda":
        try:
            import pycuda
        except ImportError:
            raise NeuralBackendError("PyCUDA specified as backend but not installed")

        try:
            import pycuda.autoinit
        except pycuda.driver.RuntimeError as e:
            raise NeuralBackendError("PyCUDA AutoInit Failed") from e
        except Exception as e:
            raise NeuralBackendError("PyCUDA AutoInit Failed with unknown error") from e
        CUDA = True
        INITIALIZED = True
        return

    if backend == "cupy":
        try:
            import cupy as cp
        except ImportError as e:
            raise NeuralBackendError(
                "CuPy specified as backend but not installed"
            ) from e
        except Exception as e:
            raise NeuralBackendError("CuPy Import Failed") from e

        try:
            cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as e:
            warn(
                NeuralBackendWarning(
                    f"CuPy initialization failed, forcing CPU mode. {e}"
                )
            )
            CUDA = False
            INITIALIZED = True
        except Exception as e:
            raise NeuralBackendError("CuPy Runtime Check failed") from e
        else:
            CUDA = True
            INITIALIZED = True
        return


def neural_initialized(func):
    """Decorator to enforce neural initialization"""

    def function(*args, **kwargs):
        if not INITIALIZED:
            raise NeuralBackendError(
                "Neural Not Initialized, call neural.init(backend='') first."
            )
        return func(*args, **kwargs)

    return function
