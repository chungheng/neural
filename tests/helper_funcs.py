#!/usr/bin/env python

"""Tests for `compneuro` package."""

import pytest
import numpy as np
from neural.utils.signal import average_snr  # pylint:disable=no-name-in-module
from scipy import signal


def assert_snr(signal1, signal2, threshold: float = 30.0):
    _asnr = average_snr(signal1, signal2, err_bias=1e-15)
    if _asnr >= threshold:
        return None
    else:
        raise AssertionError(
            f"SNR of {_asnr:.3f} is not greater than threshold {threshold:.3f}\n"
            f"signal1: {signal1}\n"
            f"signal2: {signal2}"
        )
