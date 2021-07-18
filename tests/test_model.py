import pytest
import numpy as np
from neural.config import cuda_available
from neural.basemodel import Model
from neural import logger


class DummyModel(Model):
    """Dummy Model that passes input to output"""

    Default_Params = dict()
    Default_States = dict(x=0.0)

    def ode(self, inp=0.0):
        self.d_x = inp


def test_model_construction():
    dum = DummyModel()
    assert dum.params == {}
    assert dum.states == {"x": 0.0}
    assert dum.initial_states == {"x": 0.0}
    assert dum.gstates == {"x": 0.0}
    assert dum.bounds == {}
    assert dum.solver.__name__ == "forward_euler"
    assert dum.callbacks == []


def test_model_compilation():
    dum = DummyModel()
    dum.compile(
        backend="scalar",
    )

    assert dum.num is None

    dum.compile(backend="numpy", num=1)

    dum.compile(backend="cuda", num=1)
    with pytest.raises(logger.NeuralBackendError, match="Unexpected backend .*"):
        dum.compile(backend="wrong")
