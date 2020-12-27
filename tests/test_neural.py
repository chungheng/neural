import pytest
import importlib


def reload_neural():
    neural = importlib.import_module("neural")
    neural = importlib.reload(neural)  # force reload to reset global parameters
    return neural


def test_decorator():
    neural = reload_neural()

    @neural.neural_initialized
    def this_should_fail():
        pass

    with pytest.raises(neural.logger.NeuralBackendError):
        this_should_fail()
    neural.init(backend="numpy")

    @neural.neural_initialized
    def this_should_pass():
        pass

    this_should_pass()


def test_init():
    neural = reload_neural()
    assert neural.INITIALIZED == False
    assert neural.CUDA == False
    neural.init(backend="numpy")
    assert neural.CUDA == False
    assert neural.INITIALIZED == True
