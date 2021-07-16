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
import typing as tp
import struct
import zlib
from binascii import unhexlify
import numpy as np
from scipy.signal import butter, lfilter
from . import errors as err


def chunk(btype, data):
    return (
        struct.pack(">I", len(data))
        + btype
        + data
        + struct.pack(">I", zlib.crc32(btype + data))
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
        support: two time points at which the stimulus starts and ends.
        amplitude: the amplitudes of the stimuli.
        sigma: variance of zero-mean Gaussian noise added to the waveform.
        ratio: a real number between 0,1 that indicates the point within
            :code:`support` where stimulus reaches the peak. Only applicable
            if :code:`mode` is either `"parabola"` or `"ramp"`.
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
        ratio = kwargs.pop("ratio", 0.9)

        start = int(support[0] // d_t)
        stop = int(support[1] // d_t)
        peak = int((1.0 - ratio) * start + ratio * stop)

        for wav, amp in zip(waveforms, amplitude):
            wav[start:peak] = np.linspace(0.0, amp, peak - start)
            wav[peak:stop] = np.linspace(amp, 0.0, stop - peak)

    def _generate_parabola(waveforms, d_t, support, amplitude, **kwargs):
        """
        Generate a set of parabolic stimuli.

        keyword arguments:
            ratio (float): a real number between 0 and 1. The point between
                `start` and `stop` where the stimulus reachs its peak.
        """
        ratio = kwargs.pop("ratio", 0.95)

        start = int(support[0] // d_t)
        stop = int(support[1] // d_t)
        peak = int((1.0 - ratio) * start + ratio * stop)

        for wav, amp in zip(waveforms, amplitude):
            wav[start:peak] = amp * np.linspace(0.0, 1, peak - start) ** 2
            wav[peak:stop] = amp * np.linspace(1, 0.0, stop - peak) ** 2

    Nt = int((duration + d_t / 2) // d_t)

    shape = (len(amplitude), Nt) if hasattr(amplitude, "__len__") else (Nt,)
    waveforms = np.zeros(shape, dtype=dtype)

    if isinstance(mode, str):
        tmp = "_generate_%s" % mode
        assert tmp in locals(), f"Stimulus type {mode} is not supported..."
        generator = locals()[tmp]

    # ad-hoc way to deal with amplitude being a scalar or a list
    if hasattr(amplitude, "__len__"):
        generator(waveforms, d_t, support, amplitude, **kwargs)
    else:
        generator([waveforms], d_t, support, [amplitude], **kwargs)

    if sigma is not None:
        waveforms += sigma * np.random.randn(*shape)
    return waveforms


def generate_spike_from_psth(
    d_t: float, psth: np.ndarray, num: int = 1, seed: int = None
) -> np.ndarray:
    """
    Generate spike sequeces from a PSTH.

    Arguments:
        d_t: the sampling interval of the input waveform.
        psth: the spike rate waveform.
        num: number of signals to generate
        seed: seed for random number generator

    Returns:
        order :code:`"C"` ndarray of type :code:`int` of shape:
            1. :code:`(num, len(psth))` if :code:`num > 1`
            2. :code:`(len(psth), )` if :code:`num == 1`
    """
    psth = np.squeeze(psth)
    if psth.ndim != 1:
        raise err.NeuralUtilityError(
            f"Only 1D PSTH is currently accepted, got {psth.ndim} instead"
        )

    rng = np.random.RandomState(seed)
    shape = (len(psth), num)
    spikes = np.zeros(shape, dtype=int, order="C")

    randvar = rng.rand(*shape)
    spikes = randvar < d_t * psth[:, None]

    return np.squeeze(spikes).T


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

    start = np.arange(0.0, d_t * len(spikes) - window, interval) // d_t
    stop = np.arange(window, d_t * len(spikes) - d_t, interval) // d_t
    start = start.astype(int, copy=False)
    stop = stop.astype(int, copy=False)

    start = start[: len(stop)]

    rates = (cum_spikes[stop] - cum_spikes[start]) / window
    stamps = np.arange(0, len(rates) * interval - d_t, interval)

    return rates, stamps


class PSTH(object):
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
        self, spikes: np.ndarray, dt: float, window: float = 20e-3, shift: float = 10e-3
    ):
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

        duration = self.dt * len(cum_spikes)
        start = np.arange(0.0, duration - self.window, self.shift) // self.dt
        stop = np.arange(self.window, duration - self.dt, self.shift) // self.dt
        start = start.astype(int, copy=False)
        stop = stop.astype(int, copy=False)

        start = start[: len(stop)]

        rates = (cum_spikes[stop] - cum_spikes[start]) / self.window
        stamps = np.arange(0, len(rates) * self.shift - self.dt, self.shift)

        return rates, stamps

    def merge(self, others):
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
    dt: float = 1.,
    axis: int = -1,
    extra_power2: int = None,
    fftshift: bool = True
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