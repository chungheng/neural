import pytest
import pycuda.gpuarray as garray
import numpy as np
from neural.config import cuda_available
from neural.basemodel import Model
from neural.recorder import Recorder, CUDARecorder, NumpyRecorder, ScalarRecorder
from neural.network import Container, Network, Symbol
from neural import utils
from neural import logger


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

    dum = Container(DummyModel(), name="Dummy", num=1)
    assert dum.num == 1
    assert dum.name == "Dummy"
    assert isinstance(dum.x, Symbol)
    dum.latex_src
    dum.graph_src

    with pytest.raises(
        logger.NeuralContainerError, match="called with value .* not understood"
    ):
        dum = Container(DummyModel(), name="Dummy", num=1)
        dum(inp="not_Symbol_or_Input_or_Number")


def test_network_construction_compilation(single_spike_data):
    dt, dur, start, stop, amp, spike = single_spike_data
    nn = Network()
    inp = nn.input(name="Test", num=2)
    dum = nn.add(DummyModel, name="Dummy", num=2)
    dum(inp=inp)
    nn.compile()

    assert "Dummy" in nn.containers
    assert len(nn.inputs) and "Test" in nn.inputs
    assert nn._iscompiled == True
    assert nn.backend == "cuda"

    with pytest.raises(logger.NeuralNetworkError, match="Duplicate container name .*"):
        nn = Network()
        inp = nn.input(name="Test", num=2)
        dum = nn.add(DummyModel, name="Dummy", num=2)
        dum2 = nn.add(DummyModel, name="Dummy", num=2)
        dum(inp=inp)
        nn.compile()

    with pytest.warns(logger.NeuralNetworkWarning, match="Size mismatches"):
        nn = Network()
        inp = nn.input(name="Test", num=2)
        dum = nn.add(DummyModel, name="Dummy", num=1)
        dum(inp=inp)
        nn.compile()

    with pytest.raises(
        logger.NeuralNetworkCompileError,
        match="Please compile before running the network.",
    ):
        nn = Network()
        inp = nn.input(name="Test", num=2)
        dum = nn.add(DummyModel, name="Dummy", num=1)
        nn.run(dt, verbose=False)


def test_network_running(single_spike_data):
    dt, dur, start, stop, amp, spike = single_spike_data
    num = 2
    wav = utils.generate_stimulus("step", dt, dur, (start, stop), np.full((num,), amp))
    wav_g = garray.to_gpu(np.ascontiguousarray(wav.T))

    nn = Network()
    inp = nn.input(name="Test", num=2)
    dum = nn.add(DummyModel, name="Dummy", num=2)
    dum(inp=inp)
    nn.compile()
    dum.record("x")
    inp(wav_g)
    nn.run(dt, verbose=False)
    np.testing.assert_almost_equal(dum.recorder.x, wav)


def test_network_backends(single_spike_data):
    dt, dur, start, stop, amp, spike = single_spike_data
    num = 2
    wav = utils.generate_stimulus("step", dt, dur, (start, stop), np.full((num,), amp))
    wav_g = garray.to_gpu(np.ascontiguousarray(wav.T))

    # cuda backend
    nn = Network(backend="cuda")
    inp = nn.input(name="Test", num=2)
    dum = nn.add(DummyModel, name="Dummy", num=2)
    dum(inp=inp)
    nn.compile()
    dum.record("x")
    inp(wav_g)
    nn.run(dt, verbose=False)
    np.testing.assert_almost_equal(dum.recorder.x, wav)


def test_recorder(single_spike_data, multi_spike_data):
    dt, dur, start, stop, amp, spike = single_spike_data
    mdl = Container(DummyModel(), name="dummy", num=1)
    mdl.record("x")
    rec = mdl.set_recorder(steps=spike.shape[-1], rate=1)
    for tt, ss in enumerate(spike):
        mdl.obj.update(dt, inp=ss)
        rec.update(index=tt)
    np.testing.assert_equal(rec.x, spike)

    # DEBUG: This currently does not work because the container instantiation
    # is not aware of the `num` argument.
    # dt, dur, start, stop, amps, spikes = multi_spike_data
    # mdl = Container(DummyModel(), name="dummy", num=len(amps))
    # mdl.record("x")
    # rec = mdl.set_recorder(steps=spikes.shape[-1], rate=1)
    # for tt, ss in enumerate(spikes.T):
    #     mdl.obj.update(dt, inp=ss)
    #     rec.update(index=tt)
    # np.testing.assert_equal(rec.x, spikes)


@pytest.mark.skipif(not cuda_available(), reason="requires CUDA")
def test_cuda_recorder(single_spike_data):
    dt, dur, start, stop, amp, spike = single_spike_data
    spikes_g = garray.to_gpu(spike)
    mdl = Container(DummyModel(), name="dummy", num=1)
    mdl.record("x")
    rec = mdl.set_recorder(steps=spikes_g.shape[-1], rate=1)
    for tt, ss in enumerate(spikes_g):
        mdl.obj.update(dt, inp=ss)
        rec.update(index=tt)
    np.testing.assert_equal(rec.x, spikes_g.get())
    del spikes_g
