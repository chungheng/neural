import pytest
import numpy as np
from neural.basemodel import Model
from neural.backend import (
    BackendMixin,
    NumbaCPUBackendMixin,
    NumbaCUDABackendMixin,
)
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
from neural.solver.basesolver import Euler
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
    assert dum.params['a'].item() == 0.0
    assert dum.params.dtype.names == ('a',)

    assert dum.states["x"].item() == 0.0
    assert dum.states.dtype.names == ("x",)


    assert dum.initial_states["x"].item() == 0.0
    assert dum.initial_states.dtype.names == ("x",)


    assert dum.gstates["x"].item() == 0.0
    assert dum.gstates.dtype.names == ("x",)

    assert dum.bounds == {}
    assert isinstance(dum.solver, Euler)
    assert dum.callbacks == []


@pytest.mark.parametrize(
    "Backend",
    [BackendMixin, NumbaCPUBackendMixin],
)
@pytest.mark.parametrize("Model", NEURON_MODELS)
def test_default_neurons(input_signal, Model, Backend):
    dt, dur, t, waveform = input_signal
    model = Model(backend=Backend, solver=Euler)
    for i, wav in enumerate(waveform):
        model.update(dt, stimulus=wav)


@pytest.mark.parametrize(
    "Backend",
    [BackendMixin, NumbaCPUBackendMixin],
)
@pytest.mark.parametrize("solver", [Euler])  # FIXME: need to test all solvers
def test_solvers(input_signal, IAF_euler_result, Backend, solver):
    dt, dur, t, waveform = input_signal
    model = IAF(backend=Backend, solver=solver)
    res = np.zeros((model.num, len(waveform)), dtype=waveform.dtype)
    for tt, wav in enumerate(waveform):
        try:
            res[:, tt] = model.v
        except TypeError:
            model.v.get(out=res[:, tt])
        model.update(dt, stimulus=wav)
    np.testing.assert_almost_equal(res, IAF_euler_result)
