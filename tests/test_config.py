import pytest
import importlib


def reload_neural():
    neural = importlib.import_module("neural")
    neural = importlib.reload(neural)  # force reload to reset global parameters
    config = importlib.import_module("neural.config")
    config = importlib.reload(config)  # force reload to reset global parameters
    return neural


def test_decorator():
    neural = reload_neural()

    @neural.config.neural_initialized
    def this_should_fail():
        pass

    with pytest.raises(neural.errors.NeuralBackendError):
        this_should_fail()
    neural.config.init(backend="numpy")

    @neural.config.neural_initialized
    def this_should_pass():
        pass

    this_should_pass()


def test_init():
    neural = reload_neural()
    assert neural.config.INITIALIZED == False
    assert neural.config.CUDA == False
    assert neural.config.BACKEND is None
    neural.config.init(backend="numpy")
    assert neural.config.CUDA == False
    assert neural.config.INITIALIZED == True
    assert neural.config.BACKEND == "numpy"
    neural.config.init(backend="scalar")
    assert neural.config.CUDA == False
    assert neural.config.INITIALIZED == True
    assert neural.config.BACKEND == "scalar"

    if neural.config.cuda_available():
        try:
            import pycuda
        except ImportError:
            pass
        else:
            neural.config.init(backend="pycuda")
            assert neural.config.CUDA == True
            assert neural.config.INITIALIZED == True
            assert neural.config.BACKEND == "pycuda"
        try:
            import cupy
        except ImportError:
            pass
        else:
            neural.config.init(backend="cupy")
            assert neural.config.CUDA == True
            assert neural.config.INITIALIZED == True
            assert neural.config.BACKEND == "cupy"
    else:
        try:
            import cupy
        except ImportError:
            pass
        else:
            neural.config.init(backend="cupy")
            assert neural.config.CUDA == False
            assert neural.config.INITIALIZED == True
            assert neural.config.BACKEND == "cupy"

    with pytest.raises(neural.errors.NeuralBackendError):
        neural.config.init(backend="not_understood")


def test_backend_array_module():
    neural = reload_neural()
    neural.config.init(backend="numpy")
    import numpy

    assert neural.config.backend_array_module() == numpy
    neural.config.init(backend="cupy")
    import cupy

    assert neural.config.backend_array_module() == cupy
