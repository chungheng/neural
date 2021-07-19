import pytest
import numpy as np
from neural.config import cuda_available
from neural.basemodel import Model
from neural import logger
from neural import utils

if cuda_available():
    import pycuda.autoinit
    import pycuda.gpuarray as garray
from neural.model.neuron import (
    IAF,
    LeakyIAF,
    HodgkinHuxley,
    Wilson,
    Rinzel,
    ConnorStevens,
)

NEURON_MODELS = [IAF, LeakyIAF, HodgkinHuxley, Wilson, Rinzel, ConnorStevens]


@pytest.fixture
def input_signal():
    dt, dur = 1e-5, 0.2

    waveform = utils.generate_stimulus("step", dt, dur, (0.05, 0.15), 20.0)
    t = np.arange(len(waveform)) * dt
    return dt, dur, t, waveform


class DummyModel(Model):
    """Dummy Model that passes input to output"""

    Default_Params = dict(a=0.0)
    Default_States = dict(x=0.0)

    def ode(self, inp=0.0):
        self.d_x = inp


def test_model_construction():
    dum = DummyModel()
    assert dum.params == {"a": 0.0}
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

    dum = DummyModel()
    dum.compile(backend="numpy")

    dum = DummyModel()
    dum.compile(backend="cuda", num=1)
    with pytest.raises(logger.NeuralBackendError, match="Unexpected backend .*"):
        dum.compile(backend="wrong")


@pytest.mark.parametrize("Model", NEURON_MODELS)
def test_default_neurons_cpu(input_signal, Model):
    dt, dur, t, waveform = input_signal
    record = np.zeros(len(waveform))
    model = Model()
    for i, wav in enumerate(waveform):
        model.update(dt, stimulus=wav)
        record[i] = model.v


@pytest.mark.parametrize("Backend", ["cuda", "scalar", "numpy"])
@pytest.mark.parametrize("Model", NEURON_MODELS)
def test_default_neurons_compiled(input_signal, Backend, Model):
    dt, dur, t, waveform = input_signal
    waveform_g = garray.to_gpu(np.ascontiguousarray(waveform))
    record = np.zeros(len(waveform))

    model = Model()
    if Backend == "scalar":
        model.compile(backend=Backend)
    else:
        model.compile(backend=Backend, num=1)
    inp = waveform if Backend != "cuda" else waveform_g
    for i, wav in enumerate(inp):
        model.update(dt, stimulus=wav)
        record[n, i] = model.v if Backend != "cuda" else model.v.get()
    np.testing.assert_almost_equal(record, np.roll(record, 1, axis=0))


def test_solvers(input_signal):
    dt, dur, t, waveform = input_signal
    solvers = list(IAF().solver_alias.values())
    record = np.zeros((len(solvers), len(waveform)))

    for n, slv in enumerate(solvers):
        model = IAF(solver=slv)
        for i, wav in enumerate(waveform):
            model.update(dt, stimulus=wav)
            record[n, i] = model.v
    np.testing.assert_almost_equal(record, np.roll(record, 1, axis=0))
