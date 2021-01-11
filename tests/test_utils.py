import pytest
import numpy as np
from neural.utils import generate_stimulus, PSTH, generate_spike_from_psth, compute_psth


@pytest.fixture
def data():
    dt, dur, start, stop, amp = 1e-4, 2, 0.5, 1.0, 100.0
    return dt, dur, start, stop, amp


def test_stimuli(data):
    dt, dur, start, stop, amp = data
    step = generate_stimulus("step", dt, dur, (start, stop), amp)
    t = np.arange(0, dur, dt)
    step_ref = np.zeros_like(t)
    step_ref[np.logical_and(t >= start, t < stop)] = amp
    print(np.sum(step), np.sum(step_ref))
    np.testing.assert_allclose(step, step_ref)


def test_psth(data):
    dt, dur, start, stop, amp = data
    spikes = generate_stimulus("spike", dt, dur, (start, stop), np.full((100,), amp))
    psth, psth_t = PSTH(spikes, d_t=dt, window=20e-3, shift=10e-3).compute()
    psth2, psth_t2 = compute_psth(spikes, d_t=dt, window=20e-3, interval=10e-3)

    np.testing.assert_equal(psth, psth2)
    np.testing.assert_equal(psth_t, psth_t2)
    np.testing.assert_approx_equal(
        amp, psth[np.logical_and(psth_t > start, psth_t < stop)].mean(), significant=1
    )
