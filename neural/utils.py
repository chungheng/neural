"""
Utility functions for simulating the synapse and neuron models.


Methods:
    generate_stimulus: generate stimuli; currently support `step`, `ramp` and
        `parabolic` stimuli.
    generate_spike_from_psth: generate spike sequences from a PSTH.
    compute_psth: compute PSTH from a set of spike sequences.
"""

import numpy as np

def generate_stimulus(mode, d_t, duration, support, amplitude, **kwargs):
    """
    Stimuli generator

    Arguments:
        mode (str): shape of the waveform.
        d_t (float): the sampling interval for the stimuli.
        duration (float): the duration of the stimuli.
        support (list): two time points at which the stimulus starts and ends.
        amplitude (float or list): the amplitudes of the stimuli.

    Keyword Arguments:
        sigma (float): variance of zero-mean Gaussian noise added to the waveform.
        ratio (float): a
    """

    def _generate_step(waveforms, d_t, support, amplitude, **kwargs):
        """
        Generate a set of step stimuli.

        No extra keyword argument is needed.
        """
        start = int(support[0] // d_t)
        stop = int(support[1] // d_t)

        for wav, amp in zip(waveforms, amplitude):
            wav[start:stop] = amp

    def _generate_ramp(waveforms, d_t, support, amplitude, **kwargs):
        """
        Generate a set of ramp stimuli.

        keyword arguments:
            ratio (float): a real number between 0 and 1. The point between
                `start` and `stop` where the stimulus reachs its peak.
        """
        ratio = kwargs.pop('ratio', 0.9)

        start = int(support[0] // d_t)
        stop = int(support[1] // d_t)
        peak = int((1.-ratio)*start + ratio*stop)

        for wav, amp in zip(waveforms, amplitude):
            wav[start:peak] = np.linspace(0., amp, peak-start)
            wav[peak:stop] = np.linspace(amp, 0., stop-peak)

    def _generate_parabola(waveforms, d_t, support, amplitude, **kwargs):
        """
        Generate a set of parabolic stimuli.

        keyword arguments:
            ratio (float): a real number between 0 and 1. The point between
                `start` and `stop` where the stimulus reachs its peak.
        """
        ratio = kwargs.pop('ratio', 0.95)

        start = int(support[0] // d_t)
        stop = int(support[1] // d_t)
        peak = int((1.-ratio)*start + ratio*stop)

        for wav, amp in zip(waveforms, amplitude):
            wav[start:peak] = amp*np.linspace(0., 1, peak-start)**2
            wav[peak:stop] = amp*np.linspace(1, 0., stop-peak)**2

    sigma = kwargs.pop('sigma', None)
    dtype = kwargs.pop('dtype', np.float64)

    num = int(duration // d_t)

    shape = (len(amplitude), num) if hasattr(amplitude, '__len__') else num
    waveforms = np.zeros(shape, dtype=dtype)

    if isinstance(mode, str):
        tmp = '_generate_%s' % mode
        assert tmp in locals(), "Stimulus type %s is not supported..." % mode
        generator = locals()[tmp]

    # ad-hoc way to deal with amplitude being a scalar or a list
    if hasattr(amplitude, '__len__'):
        generator(waveforms, d_t, support, amplitude, **kwargs)
    else:
        generator([waveforms], d_t, support, [amplitude], **kwargs)

    if sigma is not None:
        waveforms += sigma*np.random.rand(shape)
    return waveforms

def generate_spike_from_psth(d_t, psth, **kwargs):
    """
    Generate spike sequeces from a PSTH.

    Arguments:
        d_t (float): the sampling interval of the input waveform.
        psth (darray): the spike rate waveform.

    Keyword Arguments:
        num (int):

    """
    num = kwargs.pop('num', 1)

    shape = (len(psth), num) if num > 1 else len(psth)
    spikes = np.zeros(shape, dtype=int, order='C')

    for i, rate in enumerate(psth):
        spikes[i] = np.random.rand(num) < d_t*rate

    return spikes.T

def compute_psth(spikes, d_t, window, interval):
    """
    Compute the peri-stimulus time histogram.

    Arguments:
        spikes (ndarray): spike sequences.
        d_t (float): time step.
        window (float): the size of the window.
        interval (float): the time shift between two consecutive windows.

    Returns:
        rates (ndarray): the average spike rate for each windows.
        stamps (ndarray): the time stamp for each windows.
    """

    if len(spikes.shape) > 1:
        axis = int(spikes.shape[0] > spikes.shape[1])
        spikes = np.mean(spikes, axis=axis)

    cum_spikes = np.cumsum(spikes)

    start = np.arange(0., d_t*len(spikes)-window, interval) // d_t
    stop = np.arange(window, d_t*len(spikes)-d_t, interval) // d_t
    start = start.astype(int, copy=False)
    stop = stop.astype(int, copy=False)

    start = start[:len(stop)]

    rates = (cum_spikes[stop] - cum_spikes[start]) / window
    stamps = np.arange(0, len(rates)*interval-d_t, interval)

    return rates, stamps

class PSTH(object):
    def __init__(self, spikes, dt, window=20e-3, shift=10e-3):
        self.window = window
        self.shift = shift
        self.dt = dt
        self.spikes = spikes

        self.psth, self.t = self.compute()

    def compute(self):
        spikes = self.spikes
        if len(spikes.shape) > 1:
            axis = int(spikes.shape[0] > spikes.shape[1])
            spikes = np.mean(spikes, axis=axis)

        cum_spikes = np.cumsum(spikes)

        duration = self.dt*len(cum_spikes)
        start = np.arange(0., duration-self.window, self.shift) // self.dt
        stop  = np.arange(self.window, duration-self.dt, self.shift ) // self.dt
        start = start.astype(int, copy=False)
        stop  = stop.astype(int, copy=False)

        start = start[:len(stop)]

        rates = (cum_spikes[stop] - cum_spikes[start]) / self.window
        stamps = np.arange(0, len(rates)*self.shift-self.dt, self.shift)

        return rates, stamps

    def merge(self, others):
        if not hasattr(others, '__len__'):
            others = [others]
        for other in others:
            assert np.all(self.t == other.t)

        stack = [self.psth]
        for other in others:
            stack.append(other.psth)

        self.psth = np.vstack(stack)
