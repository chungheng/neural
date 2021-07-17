import pytest
import pycuda.gpuarray as garray
import numpy as np
from neural.config import cuda_available
from neural.basemodel import Model
from neural.recorder import Recorder, CUDARecorder, NumpyRecorder, ScalarRecorder
from neural.network import Container
from neural import utils


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


def test_recorder(single_spike_data, multi_spike_data):
    dt, dur, start, stop, amp, spike = single_spike_data
    mdl = Container(DummyModel(), name="dummy", num=1)
    mdl.record("x")
    rec = mdl.set_recorder(steps=spike.shape[-1], rate=1)
    for tt, ss in enumerate(spike):
        mdl.obj.update(dt, inp=ss)
        rec.update(index=tt)
    np.testing.assert_equal(rec.x, spike)

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
    np.testing.assert_equal(rec.x.get(), spikes_g.get())
    del spikes_g
