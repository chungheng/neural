# pylint:disable=no-member
"""Test Utility Module of CompNeuro

Tests:
    1. Plot Submodule
    2. Signal Submodule
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
import neural.errors as err

import pytest
import numpy as np
from neural import utils  # pylint:disable=import-error

DT = 1e-4
DUR = 1.0
T = np.arange(0, DUR, DT)


@pytest.fixture
def data():
    dt = 1e-4
    dur = 1.0
    t = np.arange(0, 1.0, dt)
    bw = 100  # 30 Hz
    num = 2
    start, stop = 0.2, 0.8
    amp = 10.0
    seed = 0
    return dt, dur, t, bw, num, start, stop, amp, seed


def test_stimuli(data):
    dt, dur, t, bw, num, start, stop, amp, seed = data
    step = utils.generate_stimulus("step", dt, dur, (start, stop), amp)
    t = np.arange(0, dur, dt)
    step_ref = np.zeros_like(t)
    step_ref[np.logical_and(t >= start, t < stop)] = amp
    print(np.sum(step), np.sum(step_ref))
    np.testing.assert_allclose(step, step_ref)


@pytest.fixture
def spikes(data):
    dt, dur, t, bw, num, start, stop, amp, seed = data

    rate = 100
    randvar = np.random.rand(len(t), num)
    spikes = randvar < rate * dt
    return rate, spikes


def test_generate_stimulus(data):
    dt, dur, t, bw, num, start, stop, amp, seed = data
    step = utils.generate_stimulus("step", dt, dur, (0.2 * dur, 0.8 * dur), 100)
    para = utils.generate_stimulus("parabola", dt, dur, (0.2 * dur, 0.8 * dur), 100)
    ramp = utils.generate_stimulus("ramp", dt, dur, (0.2 * dur, 0.8 * dur), 100)

    assert len(step) == len(t)
    assert len(para) == len(t)
    assert len(ramp) == len(t)
    assert step.max() == 100
    assert para.max() == 100
    assert ramp.max() == 100

    amps = np.linspace(1, 100, num)
    step = utils.generate_stimulus("step", dt, dur, (0.2 * dur, 0.8 * dur), amps)
    para = utils.generate_stimulus("parabola", dt, dur, (0.2 * dur, 0.8 * dur), amps)
    ramp = utils.generate_stimulus("ramp", dt, dur, (0.2 * dur, 0.8 * dur), amps)

    assert step.shape == (len(amps), len(t))
    assert para.shape == (len(amps), len(t))
    assert ramp.shape == (len(amps), len(t))

    np.testing.assert_equal(step.max(axis=1), amps)
    np.testing.assert_equal(para.max(axis=1), amps)
    np.testing.assert_equal(ramp.max(axis=1), amps)


def test_compute_psth(data, spikes):
    dt, dur, t, bw, num, start, stop, amp, seed = data
    rate, ss = spikes
    psth, psth_t = utils.compute_psth(ss, dt, window=2e-2, interval=1e-2)
    assert np.abs(np.mean(psth) - rate) / rate < 0.2


def test_snr(data):
    dt, dur, t, bw, num, start, stop, amp, seed = data

    amps = np.arange(0, 100, num)
    step = utils.generate_stimulus("step", dt, dur, (0.2 * dur, 0.8 * dur), amps)
    snr_inf = utils.snr(step, step)
    assert snr_inf.shape == step.shape
    assert np.all(snr_inf == np.inf)


def test_random_signal(data):
    dt, dur, t, bw, num, start, stop, amp, seed = data
    sig = utils.random_signal(t, bw, num, seed=seed)
    assert sig.shape == (num, len(t))

    # test Power
    for v in np.mean(sig**2, axis=-1):
        assert np.abs(v - 1) < 1e-10

    # test Bandwidth

    # test seed
    sig2 = utils.random_signal(t, bw, num, seed=seed)
    np.testing.assert_equal(sig, sig2)

    # test RNG
    rng = np.random.RandomState(seed)
    sig1_1 = utils.random_signal(t, bw, num, seed=rng)
    sig1_2 = utils.random_signal(t, bw, num, seed=rng)
    sig1_3 = utils.random_signal(t, bw, num, seed=rng)
    rng = np.random.RandomState(seed)
    sig2_1 = utils.random_signal(t, bw, num, seed=rng)
    sig2_2 = utils.random_signal(t, bw, num, seed=rng)
    sig2_3 = utils.random_signal(t, bw, num, seed=rng)
    np.testing.assert_equal(sig1_1, sig2_1)
    np.testing.assert_equal(sig1_2, sig2_2)
    np.testing.assert_equal(sig1_3, sig2_3)


@pytest.fixture
def signal_data():
    dt = 1e-4
    dur = 1.0
    t = np.arange(0, 1.0, dt)
    bw = 100  # 30 Hz
    num = 2
    seed = 0
    return dt, dur, t, bw, num, seed


@pytest.fixture
def signal_spikes(signal_data):
    dt, dur, t, bw, num, seed = signal_data

    rate = 100
    randvar = np.random.rand(len(t), num)
    spikes = randvar < rate * dt
    return rate, spikes


@pytest.fixture
def matrix_data():
    return np.random.rand(100, len(T))


@pytest.fixture
def spike_data():
    spikes = np.random.rand(100, len(T)) < 0.5
    return spikes.astype(float)


def test_spikes_detect():
    volt = np.array([0.0, 1.0, 0.0, 0.0, 5.0, 2.0, 0.0])
    mask_ref = np.array([False, True, False, False, True, False, False])
    mask = utils.spike_detect(volt)
    np.testing.assert_equal(mask_ref, mask)

    volt = np.array([0.0, 1.0, 0.0, 0.0, 5.0, 2.0, 0.0])
    mask_ref = np.array([False, False, False, False, True, False, False])
    mask = utils.spike_detect(volt, threshold=2.0)
    np.testing.assert_equal(mask_ref, mask)

    volt = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 5.0, 2.0, 0.0],
            [0.0, 0.0, 5.0, 0.0, 0.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        ]
    )
    mask_ref_row = np.array(
        [
            [False, True, False, False, True, False, False],
            [False, False, True, False, False, True, False],
            [False, False, False, False, False, True, False],
        ]
    )
    mask_ref_col = np.array(
        [
            [False, False, False, False, False, False, False],
            [False, False, True, False, False, False, True],
            [False, False, False, False, False, False, False],
        ]
    )
    np.testing.assert_equal(mask_ref_row, utils.spike_detect(volt, axis=-1))
    np.testing.assert_equal(mask_ref_row, utils.spike_detect(volt, axis=1))
    np.testing.assert_equal(mask_ref_col, utils.spike_detect(volt, axis=0))


@pytest.mark.parametrize("t_start", [0.0, -0.1, 0.1])
def test_convolve(t_start):
    dt = 1e-3

    # both t and t_filt are even number
    t = np.arange(t_start, t_start + 0.5, dt)
    t_filt = np.arange(0, 0.1, dt)
    u = utils.generate_stimulus("step", dt, 0.5, (0.1, 0.4), 10.0, 5.0)
    h = utils.generate_stimulus("step", dt, 0.1, (0, 0.1), 0.0, 1.0)

    N_t = dict(
        full=len(t) + len(t_filt) - 1, same=len(t), valid=len(t) - len(t_filt) + 1
    )

    for method in ["auto", "fft", "direct"]:
        for mode in ["full", "same", "valid"]:
            t_out, hu = utils.convolve(u, h, method=method, mode=mode)
            assert hu.shape == (len(t_out),) == (N_t[mode],)

    # test time_vector
    t_start_all = dict(full=t_start, same=t_start + 0.05 - dt, valid=t_start + 0.1 - dt)

    for method in ["auto", "fft", "direct"]:
        for mode in ["full", "same", "valid"]:
            t_out, hu = utils.convolve(
                u, h, dt, t_u=t, t_h=t_filt, method=method, mode=mode
            )
            assert np.isclose(t_out[0], t_start_all[mode])
            assert hu.shape == (len(t_out),) == (N_t[mode],)

    # t is even number long and t_filt is odd number long
    t = np.arange(t_start, t_start + 0.5, dt)
    t_filt = np.arange(0, 0.1 + dt, dt)
    u = utils.generate_stimulus("step", dt, 0.5, (0.1, 0.4), 10.0, 5.0)
    h = utils.generate_stimulus("step", dt, 0.1 + dt, (0, 0.1), 0.0, 1.0)

    N_t = dict(
        full=len(t) + len(t_filt) - 1, same=len(t), valid=len(t) - len(t_filt) + 1
    )

    t_start_all = dict(full=t_start, same=t_start + 0.05, valid=t_start + 0.1)

    for method in ["auto", "fft", "direct"]:
        for mode in ["full", "same", "valid"]:
            t_out, hu = utils.convolve(
                u, h, dt, t_u=t, t_h=t_filt, method=method, mode=mode
            )
            assert np.isclose(
                t_out[0], t_start_all[mode]
            ), f"{mode}, {t_out[0]}, {t_start_all[mode]}"
            assert hu.shape == (len(t_out),) == (N_t[mode],)

    # batched process
    t = np.arange(t_start, t_start + 0.5, dt)
    t_filt = np.arange(0, 0.1 + dt, dt)
    u = utils.generate_stimulus("step", dt, 0.5, (0.1, 0.4), np.full(5, 10.0), 5.0)
    h = utils.generate_stimulus("step", dt, 0.1 + dt, (0, 0.1), 0.0, 1.0)
    t_out, hu = utils.convolve(u, h, dt, t_u=t, t_h=t_filt)

    assert hu.shape == (5, len(t_out))
    assert t_out.ndim == 1
