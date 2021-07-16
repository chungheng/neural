import pytest
import numpy as np
from neural import utils  # pylint:disable=import-error


@pytest.fixture
def data():
    dt = 1e-4
    dur = 1.0
    t = np.arange(0, 1.0, dt)
    bw = 100  # 30 Hz
    num = 2
    start, stop = .2, .8
    amp = 10.
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


def test_psth(data):
    dt, dur, t, bw, num, start, stop, amp, seed = data
    spikes = utils.generate_stimulus("spike", dt, dur, (start, stop), np.full((100,), amp))
    psth, psth_t = utils.PSTH(spikes, d_t=dt, window=20e-3, shift=10e-3).compute()
    psth2, psth_t2 = utils.compute_psth(spikes, d_t=dt, window=20e-3, interval=10e-3)

    np.testing.assert_equal(psth, psth2)
    np.testing.assert_equal(psth_t, psth_t2)
    np.testing.assert_approx_equal(
        amp, psth[np.logical_and(psth_t > start, psth_t < stop)].mean(), significant=1
    )
    dt = 1e-4
    dur = 1.0
    t = np.arange(0, 1.0, dt)
    bw = 100  # 30 Hz
    num = 2
    seed = 0
    return dt, dur, t, bw, num, seed

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


def test_generate_spike_from_psth(data):
    dt, dur, t, bw, num, start, stop, amp, seed = data
    rate = 100
    step = utils.generate_stimulus("step", dt, dur, (0.2 * dur, 0.8 * dur), rate)
    ss = utils.generate_spike_from_psth(dt, step, num=num, seed=seed)
    assert ss.shape == (num, len(t))
    assert np.sum(ss[:, np.logical_and(t < 0.2 * dur, t > 0.8 * dur)]) == 0
    assert np.max(np.abs(np.sum(ss, axis=1) / (0.6 * dur) - rate) / rate) < 0.2

    ss = utils.generate_spike_from_psth(dt, step, num=1, seed=seed)
    assert ss.shape == (len(t),)


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
    for v in np.mean(sig ** 2, axis=-1):
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
