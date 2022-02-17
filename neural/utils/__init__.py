"""Utilities of Neural
"""
from .signal import (
    create_rng,
    generate_stimulus,
    generate_spike_from_psth,
    compute_psth,
    snr,
    average_snr,
    random_signal,
    nextpow2,
    fft,
    spike_detect,
    spike_detect_local,
    convolve,
)

from . import plot
from . import model
from .array import *
