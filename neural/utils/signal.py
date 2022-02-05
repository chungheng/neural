"""Signal Generation & Analysis utilities

Provides the following methods:

- :func:`generate_stimulus`: generate stimuli, currently support :code:`step`,
  :code:`ramp` and :code:`parabolic` stimuli.
- :func:`generate_spike_from_psth`: generate spike sequences from a PSTH.
- :func:`compute_psth`: compute PSTH from a set of spike sequences.
- :func:`snr`: compute Signal-to-Noise-Ratio between a signal and it\'s noisy
  version in deciBel at each data point
- :func:`average_snr`: compute SNR for entire signals
- :func:`fft`: compute Fourier Transform of given signal(s), and returns the
  spectrum as well as the frequency vector
- :func:`nextpow2`: compute next smallest power of 2 exponent for given
  number, same as :code:`nextpow2` in MATLAB.
- :func:`spike_detect`: detect spike in 1D/2D variable arrays and return
  binary array as mask for where a spike has occured.
"""

import typing as tp
from warnings import warn
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
from scipy.signal import convolve as scipy_convolve
from numbers import Number
import typing as tp
import numpy as np
import numpy.typing as npt
from .. import errors as err

# pylint:disable=no-member
def create_rng(seed: tp.Union[int, np.random.RandomState]):
    if isinstance(seed, np.random.RandomState):
        return seed
    return np.random.RandomState(seed)


# pylint:enable=no-member


def generate_stimulus(
    mode: str,
    d_t: float,
    duration: float,
    support: tp.Tuple[float, float],
    amplitude: tp.Union[float, np.ndarray],
    sigma: float = None,
    dtype: npt.DTypeLike = np.float64,
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
        waveforms += sigma * np.random.rand(*shape)
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
        order :code:`"C"` ndarray of type :code:`int` of shape

            1. :code:`(num, len(psth))` if :code:`num > 1`
            2. :code:`(len(psth), )` if :code:`num == 1`
    """
    psth = np.squeeze(psth)
    if psth.ndim != 1:
        raise err.NeuralUtilityError(
            f"Only 1D PSTH is currently accepted, got {psth.ndim} instead"
        )

    rng = np.random.RandomState(seed)  # pylint:disable=E1101
    shape = (len(psth), num)
    spikes = np.zeros(shape, dtype=int, order="C")

    randvar = rng.rand(*shape)
    spikes = randvar < d_t * psth[:, None]

    return np.squeeze(spikes).T


def compute_psth(
    spikes: np.ndarray, d_t: float, window: float, interval: float
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Compute the peri-stimulus time histogram.

    Arguments:
        spikes: spike sequences.
        d_t: time step.
        window: the size of the window.
        interval: the time shift between two consecutive windows.

    Returns:
        A 2-tuple of

            1. the average spike rate for each windows.
            2. the time stamp for each windows.
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


def snr(u: np.ndarray, u_rec: np.ndarray, err_bias: float = 0.0) -> np.ndarray:
    """Compute Signal to Noise Ratio at Every Data Point

    Computes the SNR according to the formula
    :math:`SNR[u, u_{rec}](t) = 10\log_{10} \\frac{u(t)^2}{err_{bias} + (u(t)-u_{rec}(t))^2}`

    Parameters:
        u: Clean Signal
        u_rec: Noisy Signal
        err_bias: bias term to be added to the denonimator of SNR term to
            avoid numerical error. Default as `0.` which results in infinite
            SNR for points where there is no error.

    Returns:
        Signal-to-Noise-Ratio in deciBel

    .. seealso:: :func:`average_snr`
    """
    _err = u - u_rec
    _snr = np.full(_err.shape, np.inf, dtype=u.dtype)
    mask = _err != 0
    _snr[mask] = 10 * np.log10(u[mask] ** 2 / _err[mask] ** 2)
    return _snr


def average_snr(u: np.ndarray, u_rec: np.ndarray, err_bias: float = 0.0) -> float:
    """Compute Avarege Signal to Noise Ratio

    Compute SNR of the entire signal instead of at every point as
    :math:`SNR[u, u_{rec}] = 10\log_{10} \\frac{E[u^2]}{err_{bias} + E[(u-u_{rec})^2]}`

    Parameters:
        u: Clean Signal
        u_rec: Noisy Signal
        err_bias: bias term to be added to the denonimator of SNR term to
            avoid numerical error. Default as `0.` which results in infinite
            SNR for points where there is no error.

    Returns:
        Scalar value Signal-to-Noise-Ratio in deciBel

    .. seealso:: :func:`snr`
    """
    _err = u - u_rec
    _err_pow = np.mean(_err**2)
    _sig_pow = np.mean(u**2)
    return 10 * np.log10(_sig_pow / (err_bias + _err_pow))


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
    if isinstance(seed, np.random.RandomState):  # pylint:disable=E1101
        rng = seed
    else:
        rng = np.random.RandomState(seed)  # pylint:disable=E1101

    wn = rng.randn(num, len(t))
    if bw is None:
        return wn
    fs = 1 / (t[1] - t[0])
    b, a = butter(5, bw, btype="low", analog=False, fs=fs)
    sig = lfilter(b, a, wn, axis=-1)
    sig_pow = np.mean(sig**2, axis=-1)
    sig /= np.sqrt(sig_pow)[:, None]  # RMS power normalization
    return sig


def nextpow2(n: "Number") -> float:
    """Find Minimum Power 2 Exponent"""
    return np.ceil(np.log2(n))


def fft(
    signal: np.ndarray,
    axis: int = -1,
    extra_power2: int = None,
    fftshift: bool = True,
    fs: float = None,
    dt: float = None,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Compute Spectrum of Signal

    Parameters:
        signal: Signal to take fft of.
        axis: axis long which to take fft, default to last
        extra_power2: extra power of 2 to add to fft when computing NFFT.
            setting it to :code:`None` will not pad the signal.
        fftshift: seeting to :code:`True` (default) will shift the signal to be
            centered at 0 Hz.
        fs: sampling frequency of the signal
        dt: temporal resolution of the signal

    .. note::

        Sampling Frequency :code:`fs` and temporal resolution :code:`dt` are
        reciprocals of each other.

    Raises:
        err.NeuralUtilityError: Raised if :code:`fs` and :code:`dt` are both
            specified but are not reciprocals or each other.


    Returns:
        A 2-tuple of frequency (in Hz) and Complex spectrum both centered
        at 0 Hz.
    """
    if fs is None and dt is None:
        dt = 1.0
    elif fs is not None and dt is not None:
        if fs != 1.0 / dt:
            raise err.NeuralUtilityError(
                f"Sampling Frequency fs={fs} but temporal resolution dt={dt}, "
                "they must be reciprocals of each other"
            )
    elif fs is not None and dt is None:
        dt = 1.0 / fs

    Nt = signal.shape[axis]
    if extra_power2 is None:
        nfft = signal.shape[axis]
    else:
        nfft = 2 ** int(nextpow2(Nt) + extra_power2)
    spec = np.fft.fft(signal, n=nfft, axis=axis)
    freq = np.fft.fftfreq(nfft, d=dt * (Nt / nfft))
    if fftshift:
        spec = np.fft.fftshift(spec, axes=axis)
        freq = np.fft.fftshift(freq, axes=axis)
    return freq, spec


def spike_detect(voltage: np.ndarray, axis=-1, **find_peaks_args) -> np.ndarray:
    """Detect Spikes from voltage trace

    Arguments:
        voltage: 1D or 2D numpy array of voltage traces.
        axis: axis along which to find spikes. Only applicable for 2D voltage.

    Keyword Arguments:
        find_peaks_args: any arguments accepted by
        :py:mod:`scipy.signal.find_peaks`

    Returns:
        A 1D or 2D binary array of same shape as input :code:`voltage`
        where :code:`True` indicates a spike

    Example:

        >>> volt = np.array([0., 1., 0., 0., 5., 2., 0.])
        >>> spk_mask = spike_detect(volt, height=1, threshold=2)
        >>> tk_idx = np.where(spk_mask)
    """
    voltage = np.asarray(voltage)
    if voltage.ndim not in [1, 2]:
        raise err.NeuralUtilityError(
            f"Input voltage array is of dimension {voltage.ndim}, expect 1 or 2"
        )

    # use scipy implementation if 1D
    if voltage.ndim == 1:
        spike_mask = np.full(voltage.shape, False, dtype=bool)
        tk_idx, _ = find_peaks(voltage, **find_peaks_args)
        spike_mask[tk_idx] = True
        return spike_mask

    # use custom implementation if 2D
    unsupported_args = list(
        set(list(find_peaks_args.keys())).intersection(
            set(["prominence", "width", "wlen", "rel_height", "plateau_size"])
        )
    )
    if len(unsupported_args) > 0:
        warn(
            (
                "2D peak finding currently does not support peaks that have "
                f"width more than 1, and arguments {unsupported_args} "
                "are not supported."
            ),
            err.NeuralUtilityWarning,
        )

    # initialize binary mask conditions
    possible_spike_mask = np.full(voltage.shape, True)
    if axis == 0:  # along column
        possible_spike_mask[[0, -1], :] = False
    else:  # along row
        possible_spike_mask[:, [0, -1]] = False

    conditions = [
        possible_spike_mask,
        np.roll(voltage, -1, axis=axis) < voltage,
        np.roll(voltage, 1, axis=axis) < voltage,
    ]

    # add height requirement if specified
    if "height" in find_peaks_args:
        if np.isscalar(find_peaks_args["height"]):
            min_height, max_height = find_peaks_args["height"], np.inf
        else:
            min_height, max_height = find_peaks_args["height"]

        conditions += [voltage >= min_height, voltage <= max_height]

    # add threshold requirement if specified
    if "threshold" in find_peaks_args:
        if np.isscalar(find_peaks_args["threshold"]):
            min_threshold, max_threshold = find_peaks_args["threshold"], np.inf
        else:
            min_threshold, max_threshold = find_peaks_args["threshold"]

        conditions += [
            voltage - np.roll(voltage, -1, axis=axis) >= min_threshold,
            voltage - np.roll(voltage, -1, axis=axis) <= max_threshold,
            voltage - np.roll(voltage, 1, axis=axis) >= min_threshold,
            voltage - np.roll(voltage, 1, axis=axis) <= max_threshold,
        ]

    spike_mask = np.logical_and.reduce(conditions)
    return spike_mask


def spike_detect_local(
    a: tp.Union[float, np.ndarray],
    b: tp.Union[float, np.ndarray],
    c: tp.Union[float, np.ndarray],
    threshold: float,
) -> tp.Union[bool, tp.Iterable[bool]]:
    """Local Spike Detector

    Checks if (b>a & b>c & b>threhsold)

    Returns:
        Either a single boolean or an ndarray of boolean, with `True` meaning
        spike
    """
    if all([np.isscalar(v) for v in [a, b, c]]):
        if (b > threshold) and (b > a) and (b > c):
            return True
        return False
    else:
        a = np.atleast_1d(a)
        b = np.atleast_1d(b)
        c = np.atleast_1d(c)
        assert a.shape == b.shape == c.shape
        return np.logical_and.reduce([b > a, b > c, b > threshold])


def convolve(
    u: np.ndarray,
    h: np.ndarray,
    dt: float = None,
    t_u: np.ndarray = None,
    t_h: np.ndarray = None,
    method: str = "auto",
    mode: str = "same",
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Convolve Two Signals Together

    This function convolves a signal and a filter and returns the
    convolution result together with a time vector that corresponds
    to the time instance of every point in the returned signal

    We first find, for convolution between two signals that start at time `t0`,
    `t1`, where `t0`, `t1` can be either negative or positive,
    the starting time of the convolution output if the full convolution output
    is returned. In this case, the total returned result has length
    :code:`len(u)+len(h)-1` in accordance with the overlap-add method of
    convolution. The very first index in this returned array corresponds to
    time `t0+t1` (regardless of the sign of `t0` and `t1`). As such, the time
    vector is first computed for the full convolution output, setting the first
    index to `t0+t1`, and finally sliced based on the :code:`mode` parameter
    selected.

    Parameters:
        u: input signal, 1D or 2D array. If 2D, assume that it's of shape
          `(N_signal, len(t_u))`
        h: filter, 1D or 2D array. If 2D, assume that it's of shape
          `(N_signal, len(t_h))`
        dt: temporal resolution of both signals
        t_u: time vector for input signal `u`
        t_h: time vector for filter `h`
        method: method of computing convolution, can be one of 'auto', 'direct'
          or 'fft'.

          .. seealso:: :py:func:`scipy.singal.convolve`

        mode: mode of convolution, can be one of 'full', 'same', 'valid'.

          .. seealso:: :py:func:`scipy.singal.convolve`

    Returns:
        t_out: an 1D array of time vector that corresponds to simulation result
        hu: Convolution result. 1D if both `h` and `u` are 1D, 2D if `u` is 2D.

    Raises:
        neural.errors.NeuralUtilityError: raised if any one of the following
          happens:

          - Shape mismatch between (`u`, `h`), (`u`, `t_u`), (`h`, `t_h`)
          - Temporal resolution mismatch between `dt`, `t_u`, `t_h`
          - Unsupoorted `mode`
    """
    u = np.atleast_2d(np.squeeze(u))
    N_sig = u.shape[0]
    h = np.atleast_2d(np.squeeze(h))
    N_filt = h.shape[0]
    if N_filt != 1:
        if N_filt != N_sig:
            raise err.NeuralUtilityError(
                f"Number of filter ({N_filt}) should either be 1 "
                f"or the same as Number of signal ({N_sig})"
            )

    # first check for all temporal resolutions
    if t_h is None:
        dt_h = None
    else:
        if t_h.ndim != 1:
            raise err.NeuralUtilityError("t_h must be 1-D")
        dt_h = t_h[1] - t_h[0]

    if t_u is None:
        dt_u = None
    else:
        if t_u.ndim != 1:
            raise err.NeuralUtilityError("t_u must be 1-D")
        dt_u = t_u[1] - t_u[0]

    # make sure all temporal resolution are the same value
    _dts = np.array([_dt for _dt in [dt, dt_h, dt_u] if _dt is not None])
    if len(_dts) > 0:
        if not np.all(np.isclose(_dts, _dts[0])):
            raise err.NeuralUtilityError(
                f"h and u must have temporal resolution equalt to dt, {_dts}"
            )
        dt = _dts[0]
    else:
        # this case is only applicable if no time information is provided
        dt = 1

    # find the starting time of the signals. The starting time is used
    # to find the appropriate time vector that corresponds to the convolution
    # output
    if t_u is None:
        t_u_start = 0.0
    else:
        assert u.shape[1] == len(t_u), "t_u must be the same length as u"
        t_u_start = t_u[0]
    if t_h is None:
        t_h_start = 0.0
    else:
        assert h.shape[1] == len(t_h), "t_h must be the same length as h"
        t_h_start = t_h[0]

    # find the starting time value of the full convolution signal
    # this corresponds to the
    t_hu_start_full = t_u_start + t_h_start

    t_full = np.arange(u.shape[1] + h.shape[1] - 1) * dt + t_hu_start_full
    if mode == "full":
        t_out = t_full
    elif mode == "same":
        # if t_out is an odd number, the slicing on both sides is exactly
        # symmetric. If t_out is an even number, slicing on the left is 1 less
        # than slicing on the right.
        start_idx = (len(t_full) - u.shape[1]) // 2
        t_out = t_full[start_idx : start_idx + u.shape[1]]
    elif mode == "valid":
        t_out = t_full[h.shape[1] - 1 : -(h.shape[1] - 1)]
    else:
        raise err.NeuralUtilityError(
            f"mode must be 'full', 'same', or 'valid', got '{mode}'"
        )

    # filter signals based on the
    if N_filt == 1:
        hu = np.vstack([scipy_convolve(_u, h[0], mode=mode, method=method) for _u in u])
    else:
        hu = np.vstack(
            [scipy_convolve(_u, _h, mode=mode, method=method) for _h, _u in zip(h, u)]
        )

    if N_sig == 1:
        return t_out, dt * hu[0]
    return t_out, dt * hu
