from turtle import back
import pytest
import numpy as np
from neural.backend import BackendMixin, NumbaCPUBackendMixin
from neural.basemodel import Model
from neural.recorder import Recorder
from neural.network import Container, Network, Symbol
from neural.solver import SOLVERS
from neural import utils
from neural import errors
import numpy as np
from helper_funcs import to_gpuarray, to_cupy


@pytest.fixture
def single_spike_data():
    dt, dur, start, stop, amp = 1e-4, 2, 0.5, 1.0, 100.0
    spike = utils.generate_stimulus("spike", dt, dur, (start, stop), amp)
    return dt, dur, start, stop, amp, spike


@pytest.fixture
def multi_spike_data():
    dt, dur, start, stop = 1e-4, 2, 0.5, 1.0
    num = 100
    amps = np.linspace(10, 100, num)
    spikes = utils.generate_stimulus("spike", dt, dur, (start, stop), amps)
    return dt, dur, start, stop, amps, spikes


class DummyModel(Model):
    """Dummy Model that passes input to output"""

    Default_Params = dict()
    Default_States = dict(x=0.0)

    def ode(self, inp=0.0):
        self.x = inp


def test_container():
    Container.isacceptable(DummyModel)
    dum = Container(DummyModel(num=1), name="Dummy")
    assert dum.num == 1
    assert dum.name == "Dummy"
    assert isinstance(dum.x, Symbol)
    dum.latex_src
    dum.graph_src

    dum = Container(DummyModel(num=10), name="Dummy")
    assert dum.num == 10
    assert dum.name == "Dummy"
    assert isinstance(dum.x, Symbol)
    dum.latex_src
    dum.graph_src

    with pytest.raises(
        errors.NeuralContainerError, match="called with value .* not understood"
    ):
        dum = Container(DummyModel(num=1), name="Dummy")
        dum(inp="not_Symbol_or_Input_or_Number")


def test_network_construction(single_spike_data):
    dt, dur, start, stop, amp, spike = single_spike_data
    nn = Network()
    inp = nn.input(name="Test", num=2)
    dum = nn.add(DummyModel, name="Dummy", num=2)
    dum(inp=inp)

    assert "Dummy" in nn.containers
    assert len(nn.inputs) and "Test" in nn.inputs

    with pytest.raises(errors.NeuralNetworkError, match="Duplicate container name: .*"):
        nn = Network()
        inp = nn.input(name="Test", num=2)
        dum = nn.add(DummyModel, name="Dummy", num=2)
        dum2 = nn.add(DummyModel, name="Dummy", num=2)
        dum(inp=inp)


@pytest.mark.parametrize("backend", [BackendMixin, NumbaCPUBackendMixin])
def test_network_running(single_spike_data, backend):
    dt, dur, start, stop, amp, spike = single_spike_data
    num = 2
    wav = utils.generate_stimulus(
        "step", dt, dur, (start, stop), np.full((num,), amp), sigma=amp
    )
    nn = Network(backend=backend)
    inp = nn.input(name="Test", num=2)
    dum = nn.add(DummyModel, name="Dummy", num=2)
    dum(inp=inp)
    dum.record("x")
    inp(np.ascontiguousarray(wav.T))
    nn.run(dt, verbose=False)
    np.testing.assert_almost_equal(dum.recorder.x, wav)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "backend", [BackendMixin]
)  # FIXME: test for Numba with new solvers too
def test_network_solvers(single_spike_data, solver, backend):
    dt, dur, start, stop, amp, spike = single_spike_data
    num = 2
    wav = utils.generate_stimulus("step", dt, dur, (start, stop), np.full((num,), amp))

    nn = Network(solver=solver, backend=backend)
    inp = nn.input(name="Test", num=num)
    dum = nn.add(DummyModel, name="Dummy", num=num)
    dum(inp=inp)
    dum.record("x")
    inp(np.ascontiguousarray(wav.T))
    nn.run(dt, verbose=False)
    np.testing.assert_almost_equal(dum.recorder.x, wav)


@pytest.mark.parametrize(
    "backend", [BackendMixin, NumbaCPUBackendMixin]
)  # FIXME: test for Numba with new solvers too
def test_recorder(single_spike_data, multi_spike_data, backend):
    dt, dur, start, stop, amp, spike = single_spike_data
    mdl = Container(DummyModel(num=1, backend=backend), name="dummy")
    mdl.record("x")
    rec = mdl.set_recorder(steps=spike.shape[-1], rate=1)
    for tt, ss in enumerate(spike):
        mdl.obj.update(dt, inp=ss)
        rec.update(index=tt)
    np.testing.assert_equal(rec.x, spike[None, :])

    mdl = Container(DummyModel(num=1, backend=backend), name="dummy")
    mdl.record("x")
    rec = mdl.set_recorder(steps=spike.shape[-1], rate=5)
    for tt, ss in enumerate(spike):
        mdl.obj.update(dt, inp=ss)
        rec.update(index=tt)
    np.testing.assert_equal(rec.x, spike[None, ::5])

    # DEBUG: This currently does not work because the container instantiation
    # is not aware of the `num` argument.
    dt, dur, start, stop, amps, spikes = multi_spike_data
    mdl = Container(DummyModel(num=len(amps), backend=backend), name="dummy")
    mdl.record("x")
    rec = mdl.set_recorder(steps=spikes.shape[-1], rate=1)
    for tt, ss in enumerate(spikes.T):
        mdl.obj.update(dt, inp=ss)
        rec.update(index=tt)
    np.testing.assert_equal(rec.x, spikes)


@pytest.mark.parametrize("conversion_f", [to_cupy, to_gpuarray])
def test_cuda_recorder(single_spike_data, conversion_f):
    dt, dur, start, stop, amp, spike = single_spike_data
    spikes_g = conversion_f(spike)
    mdl = Container(DummyModel(num=1), name="dummy")
    mdl.record("x")
    rec = mdl.set_recorder(steps=spikes_g.shape[-1], rate=1)
    for tt, ss in enumerate(spikes_g):
        mdl.obj.update(dt, inp=ss)
        rec.update(index=tt)
    np.testing.assert_equal(rec.x, utils.array.cudaarray_to_cpu(spikes_g)[None, :])
