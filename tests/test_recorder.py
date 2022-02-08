import pytest
import numpy as np

from neural.recorder import Recorder
from neural.utils.array import cudaarray_to_cpu

import numpy as np
import sys

to_cupy = None
to_gpuarray = None
try:
    import cupy

    to_cupy = lambda x: cupy.asarray(x)
except:
    pass
try:
    import pycuda.autoprimaryctx
    from pycuda import gpuarray

    to_gpuarray = lambda x: gpuarray.to_gpu(x)
except:
    pass


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
            arr_spk = np.random.randn(steps).astype(np.int_)
        else:
            arr_spk = np.random.randn(steps, num).astype(np.int_)
        arr_spk[arr_spk > 0] = 1
        arr_spk[arr_spk <= 0] = 0
        self.arr_spk = arr_spk
        self.attr_spike = self.arr_spk[0]

    def to_gpu(self, conv_func):
        self.arr = conv_func(self.arr)
        self.arr_spk = conv_func(self.arr_spk)
        self.attr = self.arr[0]
        self.attr_spike = self.arr_spk[0]

    def to_cpu(self):
        cudaarray_to_cpu(self.arr, out=self.arr)
        cudaarray_to_cpu(self.arr_spk, out=self.arr_spk)
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


@pytest.mark.parametrize("conversion_f", [to_cupy, to_gpuarray])
def test_construction(data, conversion_f):
    """Test that __new__ generates the correct recorders"""
    data.to_cpu()
    rec = Recorder(data, data.attr_names(), data.steps, rate=1)
    assert rec.obj == data
    assert rec.rate == 1
    assert rec.total_steps == data.steps
    assert all([k in rec.dct for k in data.attr_names()])
    assert "attr_spike" in rec.spike_vars
    np.testing.assert_equal(rec.attr, rec.dct["attr"])

    if conversion_f is not None:
        data.to_gpu(conversion_f)
    rec = Recorder(data, data.attr_names(), data.steps, rate=1, gpu_bufsize=10)
    assert rec.gpu_buf
    assert rec.gpu_bufsize == 10
    for attr in data.attr_names():
        assert attr in rec.gpu_buf
        assert rec.gpu_buf[attr].shape == (data.num, rec.gpu_bufsize)


@pytest.mark.parametrize("conversion_f", [to_cupy, to_gpuarray])
def test_continous_recording(data, conversion_f):
    # Continous data in CPU
    rec = Recorder(data, ["attr"], data.steps, rate=1)
    rec2 = Recorder(data, ["attr"], data.steps, rate=1)
    for tt in range(data.steps):
        data.update(tt)
        rec.update(tt)
        rec2.update()
    np.testing.assert_equal(rec.attr, data.arr.T)
    np.testing.assert_equal(rec.attr, rec2.attr)

    rate = 3
    rec = Recorder(data, ["attr"], data.steps, rate=rate)
    rec2 = Recorder(data, ["attr"], data.steps, rate=rate)
    for tt in range(data.steps):
        data.update(tt)
        rec.update()
        rec2.update(tt)
    np.testing.assert_equal(rec.attr, data.arr[::rate].T)
    np.testing.assert_equal(rec.attr, rec2.attr)

    # Continous data in GPU
    data.to_gpu(conversion_f)
    rec = Recorder(data, ["attr"], data.steps, rate=1)
    rec2 = Recorder(data, ["attr"], data.steps, rate=1)
    for tt in range(data.steps):
        data.update(tt)
        rec.update(tt)
        rec2.update(tt)
    np.testing.assert_equal(rec.attr, cudaarray_to_cpu(data.arr).T)
    np.testing.assert_equal(rec.attr, rec2.attr)

    rate = 3
    rec = Recorder(data, ["attr"], data.steps, rate=rate)
    rec2 = Recorder(data, ["attr"], data.steps, rate=rate)
    for tt in range(data.steps):
        data.update(tt)
        rec.update()
        rec2.update(tt)
    np.testing.assert_equal(rec.attr, cudaarray_to_cpu(data.arr)[::rate].T)
    np.testing.assert_equal(rec.attr, rec2.attr)

    # Continous Scalar data in CPU
    data = data.to_scalar()
    rec = Recorder(data, ["attr"], data.steps, rate=1)
    rec2 = Recorder(data, ["attr"], data.steps, rate=1)
    for tt in range(data.steps):
        data.update(tt)
        rec.update()
        rec2.update(tt)
    np.testing.assert_equal(rec.attr, data.arr)
    np.testing.assert_equal(rec.attr, rec2.attr)

    rate = 3
    rec = Recorder(data, ["attr"], data.steps, rate=rate)
    rec2 = Recorder(data, ["attr"], data.steps, rate=rate)
    for tt in range(data.steps):
        data.update(tt)
        rec.update()
        rec2.update(tt)
    np.testing.assert_equal(rec.attr, data.arr[::rate].T)
    np.testing.assert_equal(rec.attr, rec2.attr)


@pytest.mark.parametrize("conversion_f", [to_cupy, to_gpuarray])
def test_spike_recording(data, conversion_f):
    # Spiking data in CPU
    rec = Recorder(data, ["attr_spike"], data.steps, rate=1)
    rec2 = Recorder(data, ["attr_spike"], data.steps, rate=1)
    for tt in range(data.steps):
        data.update(tt)
        rec.update()
        rec2.update(tt)
    np.testing.assert_array_equal(rec.attr_spike, data.arr_spk.T)
    np.testing.assert_array_equal(rec2.attr_spike, data.arr_spk.T)

    rate = 3
    rec = Recorder(data, ["attr_spike"], data.steps, rate=rate)
    rec2 = Recorder(data, ["attr_spike"], data.steps, rate=rate)
    accumulated_spikes = np.zeros((data.num, rec.steps), dtype=np.int_, order="F")
    for tt in range(data.steps):
        data.update(tt)
        rec.update()
        rec2.update(tt)
        accumulated_spikes[..., int(tt / rate)] += data.attr_spike
    np.testing.assert_array_equal(rec.attr_spike, accumulated_spikes)
    np.testing.assert_array_equal(rec2.attr_spike, rec2.attr_spike)

    # Spiking data in GPU
    data.to_gpu(conversion_f)
    rec = Recorder(data, ["attr_spike"], data.steps, rate=1)
    rec2 = Recorder(data, ["attr_spike"], data.steps, rate=1)
    for tt in range(data.steps):
        data.update(tt)
        rec.update()
        rec2.update(tt)
    np.testing.assert_equal(rec.attr_spike, cudaarray_to_cpu(data.arr_spk).T)
    np.testing.assert_equal(rec2.attr_spike, cudaarray_to_cpu(data.arr_spk).T)

    rate = 3
    rec = Recorder(data, ["attr_spike"], data.steps, rate=rate)
    rec2 = Recorder(data, ["attr_spike"], data.steps, rate=rate)
    accumulated_spikes = np.zeros((data.num, rec.steps), dtype=np.int_, order="F")
    for tt in range(data.steps):
        data.update(tt)
        rec.update()
        rec2.update(tt)
        assert tt == rec2.curr_step == rec.curr_step
        try:
            np.testing.assert_array_equal(
                cudaarray_to_cpu(rec2.gpu_buf["attr_spike"]),
                cudaarray_to_cpu(rec.gpu_buf["attr_spike"]),
            )
        except:
            print(tt)
        accumulated_spikes[..., int(tt // rate)] += cudaarray_to_cpu(data.attr_spike)
    np.testing.assert_equal(rec.attr_spike, accumulated_spikes)
    np.testing.assert_equal(rec2.attr_spike, rec.attr_spike)

    # Scalar Spiking data in GPU
    data = data.to_scalar()
    rec = Recorder(data, ["attr_spike"], data.steps, rate=1)
    rec2 = Recorder(data, ["attr_spike"], data.steps, rate=1)
    for tt in range(data.steps):
        data.update(tt)
        rec.update()
        rec2.update(tt)
    np.testing.assert_equal(rec.attr_spike, data.arr_spk)
    np.testing.assert_equal(rec2.attr_spike, data.arr_spk)

    rate = 3
    rec = Recorder(data, ["attr_spike"], data.steps, rate=rate)
    rec2 = Recorder(data, ["attr_spike"], data.steps, rate=rate)
    accumulated_spikes = np.zeros((rec.steps,))
    for tt in range(data.steps):
        data.update(tt)
        rec.update()
        rec2.update(tt)
        accumulated_spikes[int(tt / rate)] += data.attr_spike
    np.testing.assert_equal(rec.attr_spike, accumulated_spikes)
    np.testing.assert_equal(rec2.attr_spike, accumulated_spikes)
