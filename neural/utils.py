"""
Utility functions for simulating the synapse and neuron models.


Methods:
    generate_stimulus: generate stimuli; currently support `step`, `ramp` and
        `parabolic` stimuli.
    generate_spike_from_psth: generate spike sequences from a PSTH.
    compute_psth: compute PSTH from a set of spike sequences.
"""

import struct
import zlib
import typing as tp
from binascii import unhexlify

import numpy as np
from .logger import NeuralUtilityError, SignalTypeNotFoundError


def chunk(type, data):
    return (
        struct.pack(">I", len(data))
        + type
        + data
        + struct.pack(">I", zlib.crc32(type + data))
    )


MINIMUM_PNG = (
    b"\x89PNG\r\n\x1A\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
    + chunk(b"IDAT", unhexlify(b"789c6300010000050001"))
    + chunk(b"IEND", b"")
)


def generate_stimulus(
    mode: str,
    d_t: float,
    duration: float,
    support: tp.Iterable[float],
    amplitude: tp.Union[float, tp.Iterable[float]],
    sigma: float = None,
    dtype: np.dtype = np.float64,
    **kwargs,
) -> np.ndarray:
    """
    Stimuli generator

    Arguments:
        mode (str): shape of the waveform.
        d_t (float): the sampling interval for the stimuli.
        duration (float): the duration of the stimuli.
        support (iterable): two time points at which the stimulus [start, end).
           It's not inclusive of the end.
        amplitude (float or list): the amplitudes of the stimuli.

    Keyword Arguments:
        sigma (float): variance of zero-mean Gaussian noise added to the waveform.
        ratio (float): a
    """

    def _generate_step(
        waveforms: np.ndarray,
        d_t: float,
        support: tp.Iterable[float],
        amplitude: float,
        **kwargs,
    ) -> None:
        """
        Generate a set of step stimuli.

        No extra keyword argument is needed.
        """
        start = int((support[0] + d_t / 2) // d_t)
        stop = int((support[1] + d_t / 2) // d_t)

        for wav, amp in zip(waveforms, amplitude):
            wav[start:stop] = amp

    def _generate_ramp(
        waveforms: np.ndarray,
        d_t: float,
        support: tp.Iterable[float],
        amplitude: float,
        **kwargs,
    ) -> None:
        """
        Generate a set of ramp stimuli.

        keyword arguments:
            ratio (float): a real number between 0 and 1. The point between
                `start` and `stop` where the stimulus reachs its peak.
        """
        ratio = kwargs.pop("ratio", 0.9)

        start = int((support[0] + d_t / 2) // d_t)
        stop = int((support[1] + d_t / 2) // d_t)
        peak = int((1.0 - ratio) * start + ratio * stop)

        for wav, amp in zip(waveforms, amplitude):
            wav[start:peak] = np.linspace(0.0, amp, peak - start)
            wav[peak:stop] = np.linspace(amp, 0.0, stop - peak)

    def _generate_parabola(
        waveforms: np.ndarray,
        d_t: float,
        support: tp.Iterable[float],
        amplitude: float,
        **kwargs,
    ) -> None:
        """
        Generate a set of parabolic stimuli.

        keyword arguments:
            ratio (float): a real number between 0 and 1. The point between
                `start` and `stop` where the stimulus reachs its peak.
        """
        ratio = kwargs.pop("ratio", 0.95)

        start = int((support[0] + d_t / 2) // d_t)
        stop = int((support[1] + d_t / 2) // d_t)
        peak = int((1.0 - ratio) * start + ratio * stop)

        for wav, amp in zip(waveforms, amplitude):
            wav[start:peak] = amp * np.linspace(0.0, 1, peak - start) ** 2
            wav[peak:stop] = amp * np.linspace(1, 0.0, stop - peak) ** 2

    def _generate_spike(
        waveforms: np.ndarray,
        d_t: float,
        support: tp.Iterable[float],
        amplitude: float,
        **kwargs,
    ) -> None:
        """
        Generate a set of poisson spikes.

        Note:
            The amplitude argument is used as rate in a Poisson Process
            from which the spikes are generated
        """
        start = int((support[0] + d_t / 2) // d_t)
        stop = int((support[1] + d_t / 2) // d_t)
        waveforms[:] = np.random.rand(*waveforms.shape) < amplitude[:, None] * d_t
        waveforms[:, :start] = 0
        waveforms[:, stop:] = 0

    amplitude = np.atleast_1d(amplitude)
    num = int((duration + d_t / 2) // d_t)

    shape = (len(amplitude), num)
    waveforms = np.zeros(shape, dtype=dtype)

    if isinstance(mode, str):
        tmp = "_generate_%s" % mode
        if tmp not in locals():
            msg = f"Stimulus type {mode} is not supported."
            raise SignalTypeNotFoundError(msg)
        generator = locals()[tmp]

    generator(waveforms, d_t, support, amplitude, **kwargs)

    if sigma is not None:
        waveforms += sigma * np.random.rand(*shape)

    if len(amplitude) == 1:  # for consistency with previous API
        waveforms = waveforms[0]
    return waveforms


def generate_spike_from_psth(
    d_t: float, psth: np.ndarray, psth_t: np.ndarray = None, num: int = 1
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Generate spike sequeces from a PSTH.

    Arguments:
        d_t: the sampling interval of the input waveform.
        psth: the spike rate waveform.

    Keyword Arguments:
        psth_t: time-stamps of the psth, optional. See Notes for behavior
        num: number of trials to generate

    Returns:
        A tuple of (time, spikes).
        Spikes is a binary numpy array of either shape
            1. (Nt, num) if num > 1
            2. (Nt,) if num == 1
        See Notes for definition of `Nt`

    Notes:
        1. If `psth_t` is specified:
            - it needs to have the same dimensionality as `psth`.
            - In this case, `d_t` is the desired time-step instead of the time-step of the `psth` array.
            - `Nt` is given as `np.arange(psth_t.min(), psth_t.max(), dt)`
        2. If `psth_t` is not specified:
            - `d_t` is the time-step of the `psth` array
            - `Nt` given as `len(psth)`

    Examples:
        >>> from neural import utils, plot
        >>> dt, dur, start, stop, amp = 1e-5, 2, 0.5, 1.0, 100.0
        >>> spikes = utils.generate_stimulus("spike", dt, dur, (start, stop), np.full((100,), amp))
        >>> psth, psth_t = utils.PSTH(spikes, d_t=dt, window=20e-3, shift=10e-3).compute()
        >>> tt, spikes = utils.generate_spike_from_psth(dt, psth, psth_t)
        >>> plot.plot_spikes(spikes, t=tt)
    """
    if psth.ndim > 1:
        psth = np.squeeze(psth)
        if psth.ndim > 1:
            raise NeuralUtilityError(
                f"Only 1D psth array is accepted, got shape ({psth.shape}) after squeezing"
            )

    if psth_t is not None:
        if psth_t.shape != psth.shape:
            raise NeuralUtilityError(
                f"psth_t shape ({psth_t.shape}) needs to be the same as psth shape ({psth.shape})"
            )
        t = np.arange(psth_t.min(), psth_t.max(), d_t)
        rate = np.interp(t, psth_t, psth)
    else:
        t = np.arange(len(psth)) * d_t
        rate = psth

    if num > 1:
        rate = np.repeat(psth[:, None], num, axis=-1)
        spikes = np.random.rand(len(t), num) < d_t * rate
    else:
        spikes = (
            np.random.rand(
                len(t),
            )
            < d_t * rate
        )

    return t, np.ascontiguousarray(spikes.T)


def compute_psth(
    spikes: np.ndarray, d_t: float, window: float, interval: float
) -> np.ndarray:
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

    start = np.arange(0.0, d_t * len(spikes) - window, interval) // d_t
    stop = np.arange(window, d_t * len(spikes) - d_t, interval) // d_t
    start = start.astype(int, copy=False)
    stop = stop.astype(int, copy=False)
    start = start[: len(stop)]

    rates = (cum_spikes[stop] - cum_spikes[start]) / window
    stamps = np.arange(0, len(rates) * interval - d_t, interval)

    return rates, stamps


class PSTH(object):
    def __init__(
        self,
        spikes: np.ndarray,
        d_t: float,
        window: float = 20e-3,
        shift: float = 10e-3,
    ):
        self.window = window
        self.shift = shift
        self.d_t = d_t
        self.spikes = spikes
        self.psth, self.t = self.compute()

    def compute(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        spikes = self.spikes
        if len(spikes.shape) > 1:
            axis = int(spikes.shape[0] > spikes.shape[1])
            spikes = np.mean(spikes, axis=axis)

        cum_spikes = np.cumsum(spikes)

        duration = self.d_t * len(cum_spikes)
        start = np.arange(0.0, duration - self.window, self.shift) // self.d_t
        stop = np.arange(self.window, duration - self.d_t, self.shift) // self.d_t
        start = start.astype(int, copy=False)
        stop = stop.astype(int, copy=False)

        start = start[: len(stop)]

        rates = (cum_spikes[stop] - cum_spikes[start]) / self.window
        stamps = np.arange(0, len(rates) * self.shift - self.d_t, self.shift)

        return rates, stamps

    def merge(self, others: tp.Union[np.ndarray, tp.Iterable[np.ndarray]]) -> None:
        if not hasattr(others, "__len__"):
            others = [others]
        for other in others:
            assert np.all(self.t == other.t)

        stack = [self.psth]
        for other in others:
            stack.append(other.psth)

        self.psth = np.vstack(stack)
