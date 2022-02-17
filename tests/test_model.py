import pytest
import numpy as np
from neural.basemodel import Model
from neural.backend import BackendMixin,CuPyBackendMixin,NumbaCPUBackendMixin,NumbaCUDABackendMixin
from neural import errors
from neural import utils
from neural.model.neuron import (
    IAF,
    LeakyIAF,
    HodgkinHuxley,
    Wilson,
    Rinzel,
    ConnorStevens,
)
from neural.solver.base_solver import Euler
from neural.solver import SOLVERS
from helper_funcs import to_cupy, to_gpuarray

NEURON_MODELS = [IAF, LeakyIAF, HodgkinHuxley, Wilson, Rinzel, ConnorStevens]


@pytest.fixture
def input_signal():
    dt, dur = 1e-5, 0.2

    waveform = utils.generate_stimulus("step", dt, dur, (0.05, 0.15), 20.0)
    t = np.arange(len(waveform)) * dt
    return dt, dur, t, waveform


@pytest.fixture
def IAF_euler_result(input_signal):
    dt, dur, t, waveform = input_signal
    model = IAF()
    res = np.zeros((model.num, len(waveform)), dtype=waveform.dtype)
    for tt, wav in enumerate(waveform):
        res[:, tt] = model.v
        model.update(dt, stimulus=wav)
    return res


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
    assert isinstance(dum.solver, Euler)
    assert dum.callbacks == []


@pytest.mark.parametrize("Backend", [BackendMixin,CuPyBackendMixin,NumbaCPUBackendMixin,NumbaCUDABackendMixin])
@pytest.mark.parametrize("Model", NEURON_MODELS)
def test_default_neurons(input_signal, Model, Backend):
    dt, dur, t, waveform = input_signal
    record = np.zeros(len(waveform))
    model = Model(backend=Backend)
    for i, wav in enumerate(waveform):
        model.update(dt, stimulus=wav)
        record[i] = model.v


@pytest.mark.parametrize("Model", NEURON_MODELS)
@pytest.mark.parametrize("conversion_f", [to_cupy, to_gpuarray])
def test_default_neurons_compiled(input_signal, Model, conversion_f):
    dt, dur, t, waveform = input_signal
    waveform_g = conversion_f(np.ascontiguousarray(waveform))
    # FIXME
    # record = np.zeros((len(BACKENDS), len(waveform)))

    # for n, Backend in enumerate(BACKENDS):
    #     model = Model()
    #     if Backend == "scalar":
    #         model.compile(backend=Backend)
    #     else:
    #         model.compile(backend=Backend, num=1)
    #     inp = waveform if Backend != "cuda" else waveform_g
    #     for i, wav in enumerate(inp):
    #         model.update(dt, stimulus=wav)
    #         record[n, i] = model.v if Backend != "cuda" else model.v.get()
    # np.testing.assert_almost_equal(record, np.roll(record, 1, axis=0))


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers(input_signal, IAF_euler_result, solver):
    dt, dur, t, waveform = input_signal
    model = IAF(solver=solver)
    res = np.zeros((model.num, len(waveform)), dtype=waveform.dtype)
    for tt, wav in enumerate(waveform):
        res[:, tt] = model.v
        model.update(dt, stimulus=wav)
    np.testing.assert_almost_equal(res, IAF_euler_result)
