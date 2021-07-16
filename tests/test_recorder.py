import pytest
import numpy as np

try:
    import pycuda.autoinit
    import pycuda.gpuarray as garray

    # pylint:disable=import-error
    from neural.recorder import Recorder, CUDARecorder, NumpyRecorder, ScalarRecorder

    # pylint:enable=import-error

    CUDA = True
except:
    CUDA = False


class FakeData:
    def __init__(self, steps=100, num=10):
        # number of steps
        self.steps = steps

        # number of components
        self.num = num

        # continous data
        if num == 1:
            self.arr = np.random.randn(steps)
        else:
            self.arr = np.random.randn(steps, num)
        self.attr = self.arr[0]

        # spike data
        if num == 1:
            arr_spk = np.random.randn(steps)
        else:
            arr_spk = np.random.randn(steps, num)
        arr_spk[arr_spk > 0] = 1
        arr_spk[arr_spk <= 0] = 0
        self.arr_spk = arr_spk
        self.attr_spike = self.arr_spk[0]

    def to_gpu(self):
        self.arr = garray.to_gpu(self.arr)
        self.arr_spk = garray.to_gpu(self.arr_spk)
        self.attr = self.arr[0]
        self.attr_spike = self.arr_spk[0]

    def to_cpu(self):
        if isinstance(self.arr, garray.GPUArray):
            self.arr = self.arr.get()
        if isinstance(self.arr_spk, garray.GPUArray):
            self.arr_spk = self.arr_spk.get()
        self.attr = self.arr[0]
        self.attr_spike = self.arr_spk[0]

    def update(self, index):
        self.attr = self.arr[index]
        self.attr_spike = self.arr_spk[index]

    def attr_names(self):
        return ["attr", "attr_spike"]

    def to_scalar(self):
        return self.__class__(steps=self.steps, num=1)


@pytest.fixture
def data():
    return FakeData()


@pytest.mark.skipif(CUDA == False, reason="No CUDA Available")
def test_construction(data):
    """Test that __new__ generates the correct recorders"""
    data.to_cpu()
    rec = Recorder(data, data.attr_names(), data.steps, rate=1)
    assert isinstance(rec, NumpyRecorder)
    assert rec.obj == data
    assert rec.rate == 1
    assert rec.total_steps == data.steps
    assert all([k in rec.dct for k in data.attr_names()])
    assert "attr_spike" in rec.spike_vars
    np.testing.assert_equal(rec.attr, rec.dct["attr"])

    data.to_gpu()
    rec = Recorder(data, data.attr_names(), data.steps, rate=1)
    assert isinstance(rec, CUDARecorder)

    scalar_data = data.to_scalar()
    rec = Recorder(scalar_data, scalar_data.attr_names(), data.steps, rate=1)
    assert isinstance(rec, ScalarRecorder)


@pytest.mark.skipif(CUDA == False, reason="No CUDA Available")
def test_continous_recording(data):
    # Continous data in CPU
    rec = Recorder(data, ["attr"], data.steps, rate=1)
    rec2 = Recorder(data, ["attr"], data.steps, rate=1)
    rec_obj = iter(rec)
    for tt in range(data.steps):
        data.update(tt)
        rec2.update(tt)
        next(rec_obj)
    np.testing.assert_equal(rec.attr, data.arr.T)
    np.testing.assert_equal(rec.attr, rec2.attr)

    rate = 3
    rec = Recorder(data, ["attr"], data.steps, rate=rate)
    rec2 = Recorder(data, ["attr"], data.steps, rate=rate)
    rec_obj = iter(rec)
    for tt in range(data.steps):
        data.update(tt)
        rec2.update(tt)
        next(rec_obj)
    np.testing.assert_equal(rec.attr, data.arr[::rate].T)
    np.testing.assert_equal(rec.attr, rec2.attr)

    # Continous data in GPU
    data.to_gpu()
    rec = Recorder(data, ["attr"], data.steps, rate=1)
    rec2 = Recorder(data, ["attr"], data.steps, rate=1)
    rec_obj = iter(rec)
    for tt in range(data.steps):
        data.update(tt)
        rec2.update(tt)
        next(rec_obj)
    np.testing.assert_equal(rec.attr, data.arr.get().T)
    np.testing.assert_equal(rec.attr, rec2.attr)

    rate = 3
    rec = Recorder(data, ["attr"], data.steps, rate=rate)
    rec2 = Recorder(data, ["attr"], data.steps, rate=rate)
    rec_obj = iter(rec)
    for tt in range(data.steps):
        data.update(tt)
        rec2.update(tt)
        next(rec_obj)
    np.testing.assert_equal(rec.attr, data.arr.get()[::rate].T)
    np.testing.assert_equal(rec.attr, rec2.attr)

    # Continous Scalar data in CPU
    data = data.to_scalar()
    rec = Recorder(data, ["attr"], data.steps, rate=1)
    rec2 = Recorder(data, ["attr"], data.steps, rate=1)
    rec_obj = iter(rec)
    for tt in range(data.steps):
        data.update(tt)
        rec2.update(tt)
        next(rec_obj)
    np.testing.assert_equal(rec.attr, data.arr)
    np.testing.assert_equal(rec.attr, rec2.attr)

    rate = 3
    rec = Recorder(data, ["attr"], data.steps, rate=rate)
    rec2 = Recorder(data, ["attr"], data.steps, rate=rate)
    rec_obj = iter(rec)
    for tt in range(data.steps):
        data.update(tt)
        rec2.update(tt)
        next(rec_obj)
    np.testing.assert_equal(rec.attr, data.arr[::rate].T)
    np.testing.assert_equal(rec.attr, rec2.attr)


@pytest.mark.skipif(CUDA == False, reason="No CUDA Available")
def test_spike_recording(data):
    # Spiking data in CPU
    rec = Recorder(data, ["attr_spike"], data.steps, rate=1)
    rec_obj = iter(rec)
    rec2 = Recorder(data, ["attr_spike"], data.steps, rate=1)
    for tt in range(data.steps):
        data.update(tt)
        next(rec_obj)
        rec2.update(tt)
    np.testing.assert_equal(rec.attr_spike, data.arr_spk.T)
    np.testing.assert_equal(rec2.attr_spike, data.arr_spk.T)

    rate = 3
    rec = Recorder(data, ["attr_spike"], data.steps, rate=rate)
    rec_obj = iter(rec)
    rec2 = Recorder(data, ["attr_spike"], data.steps, rate=rate)
    accumulated_spikes = np.zeros((rec.steps, data.num))
    for tt in range(data.steps):
        data.update(tt)
        next(rec_obj)
        rec2.update(tt)
        accumulated_spikes[int(tt / rate)] += data.attr_spike
    np.testing.assert_equal(rec.attr_spike, accumulated_spikes.T)
    np.testing.assert_equal(rec2.attr_spike, accumulated_spikes.T)

    # Spiking data in GPU
    data.to_gpu()
    rec = Recorder(data, ["attr_spike"], data.steps, rate=1)
    rec_obj = iter(rec)
    rec2 = Recorder(data, ["attr_spike"], data.steps, rate=1)
    for tt in range(data.steps):
        data.update(tt)
        next(rec_obj)
        rec2.update(tt)
    np.testing.assert_equal(rec.attr_spike, data.arr_spk.get().T)
    np.testing.assert_equal(rec2.attr_spike, data.arr_spk.get().T)

    rate = 3
    rec = Recorder(data, ["attr_spike"], data.steps, rate=rate)
    rec_obj = iter(rec)
    rec2 = Recorder(data, ["attr_spike"], data.steps, rate=rate)
    accumulated_spikes = garray.zeros((rec.steps, data.num), dtype=int)
    for tt in range(data.steps):
        data.update(tt)
        next(rec_obj)
        rec2.update(tt)
        accumulated_spikes[int(tt / rate)] += data.attr_spike
    np.testing.assert_equal(rec.attr_spike, accumulated_spikes.get().T)
    np.testing.assert_equal(rec2.attr_spike, accumulated_spikes.get().T)

    # Scalar Spiking data in GPU
    data = data.to_scalar()
    rec = Recorder(data, ["attr_spike"], data.steps, rate=1)
    rec_obj = iter(rec)
    rec2 = Recorder(data, ["attr_spike"], data.steps, rate=1)
    for tt in range(data.steps):
        data.update(tt)
        next(rec_obj)
        rec2.update(tt)
    np.testing.assert_equal(rec.attr_spike, data.arr_spk)
    np.testing.assert_equal(rec2.attr_spike, data.arr_spk)

    rate = 3
    rec = Recorder(data, ["attr_spike"], data.steps, rate=rate)
    rec_obj = iter(rec)
    rec2 = Recorder(data, ["attr_spike"], data.steps, rate=rate)
    accumulated_spikes = np.zeros((rec.steps,))
    for tt in range(data.steps):
        data.update(tt)
        next(rec_obj)
        rec2.update(tt)
        accumulated_spikes[int(tt / rate)] += data.attr_spike
    np.testing.assert_equal(rec.attr_spike, accumulated_spikes)
    np.testing.assert_equal(rec2.attr_spike, accumulated_spikes)
