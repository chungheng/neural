import pytest
import numpy as np
from neural.utils import generate_stimulus


def test_stimuli():
    dt, dur, start, stop, amp = 1e-4, 2, 0.5, 1.0, 100.0
    step = generate_stimulus("step", dt, dur, (start, stop), amp)
    t = np.arange(0, dur, dt)
    step_ref = np.zeros_like(t)
    step_ref[np.logical_and(t >= start, t < stop)] = amp
    print(np.sum(step), np.sum(step_ref))
    np.testing.assert_allclose(step, step_ref)


def test_psth():
    dt, dur, start, stop, amp, rate = 1e-4, 2, 0.5, 1.0, 1.0, 100.0
    spikes = generate_stimulus("spike", dt, dur, (start, stop), amp, rate=rate)
