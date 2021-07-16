"""
Utility functions for simulating the synapse and neuron models.

Methods:
    generate_stimulus: generate stimuli; currently support `step`, `ramp` and
        `parabolic` stimuli.
    generate_spike_from_psth: generate spike sequences from a PSTH.
    compute_psth: compute PSTH from a set of spike sequences.
    snr: compute Signal-to-Noise-Ratio between a signal and it's noisy version
        in deciBel
    fft: compute Fourier Transform of given signal(s), and returns the spectrum
        as well as the frequency vector
    nextpow2: compute next smallest power of 2 exponent for given number, same
        as :code:`nextpow2` in MATLAB.
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
    support: tp.Tuple[float, float],
    amplitude: tp.Union[float, np.ndarray],
    sigma: float = None,
    dtype: "dtype" = np.float64,
    **kwargs,
) -> np.ndarray:
    """
    Stimuli generator

    Arguments:
        mode: shape of the waveform.
        d_t: the sampling interval for the stimuli.
        duration: the duration of the stimuli.
        support: two time points at which the stimulus [start, end).
           It's not inclusive of the end.
        amplitude: the amplitudes of the stimuli.
        sigma: variance of zero-mean Gaussian noise added to the waveform.

    Keyword Arguments:
        ratio: a
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


class PSTH:
    """Peri-Stimulus Time Histogram

    This class facilitates computing PSTH of input spikes in an OOP fashion.
    For 2D spiking data, the PSTH is computed as an average across spike times
    for all neurons.

    .. seealso:: :py:func:`compute_psth`

    Parameters:
        spikes: 1D or 2D binary array of spike trains to compute PSTH from. For
            2D array, the temporal axis is infered as the longer dimension. The
            other dimension is treated as neuron indices.
        dt: time-step for spikes along the temporal dimension of :code:`spikes`
    """
    def __init__(
        self,
        spikes: np.ndarray,
        d_t: float,
        window: float = 20e-3,
        shift: float = 10e-3
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


def snr(u: np.ndarray, u_rec: np.ndarray) -> np.ndarray:
    """Compute Signal to Noise Ratio

    Computes the SNR according to the formula
    :math:`SNR[u, u_{rec}] = 10\log_{10} \frac{u^2}{(u-u_{rec})^2}`

    Parameters:
        u: Clean Signal
        u_rec: Noisy Signal

    Returns:
        snr: Signal-to-Noise-Ratio in deciBel
    """
    err = u - u_rec
    _snr = np.full(err.shape, np.inf, dtype=u.dtype)
    mask = err != 0
    _snr[mask] = 10 * np.log10(u[mask] ** 2 / err[mask] ** 2)
    return _snr


def random_signal(
    t: np.ndarray, bw: float = None, num: int = 1, seed: int = None
) -> np.ndarray:
    """Generate Random Signal

    Parameters:
        t: time points
        bw: bandwidth of output signal in Hz. If specified, lowpass filter white
            noise signal with butterworth filter. If :code:`None` (default),
            return white noise signal.
        num: number of signals to generate.
        seed: seed for random number generator.

    Returns:
        A float ndarray of shape :code:`(num, len(t))` that is of bandwidth
        :code:`bw`.
    """
    if isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        rng = np.random.RandomState(seed)

    wn = rng.randn(num, len(t))
    if bw is None:
        return wn
    fs = 1 / (t[1] - t[0])
    b, a = butter(5, bw, btype="low", analog=False, fs=fs)
    sig = lfilter(b, a, wn, axis=-1)
    pow = np.mean(sig ** 2, axis=-1)
    sig /= np.sqrt(pow)[:, None]  # RMS power normalization
    return sig


def nextpow2(n: "Number") -> float:
    """Find Minimum Power 2 Exponent"""
    return np.ceil(np.log2(n))


def fft(
    signal: np.ndarray,
    dt: float = 1.0,
    axis: int = -1,
    extra_power2: int = None,
    fftshift: bool = True,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Compute Spectrum of Signal

    Parameters:
        signal: Signal to take fft of.
        dt: time resolution of the signal
        axis: axis long which to take fft, default to last
        extra_power2: extra power of 2 to add to fft when computing NFFT.
            setting it to :code:`None` will not pad the signal.
        fftshift: seeting to :code:`True` (default) will shift the signal to be
            centered at 0 Hz.

    Returns:
        A 2-tuple of frequency (in Hz) and Spectrum
    """
    Nt = signal.shape[axis]
    if extra_power2 is None:
        nfft = Nt
    else:
        nfft = 2 ** int(nextpow2(Nt) + extra_power2)
    spec = np.fft.fft(signal, n=nfft, axis=axis)
    freq = np.fft.fftfreq(nfft, d=dt * (Nt / nfft))
    if fftshift:
        spec = np.fft.fftshift(spec, axes=axis)
        freq = np.fft.fftshift(freq, axes=axis)
    return freq, spec
