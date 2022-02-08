#!/usr/bin/env python

"""Tests for `compneuro` package."""

import pytest
import numpy as np
from neural.utils.signal import average_snr  # pylint:disable=no-name-in-module
from scipy import signal

cupy = None
gpuarray = None
to_cupy = None
to_gpuarray = None

try:
    import cupy
    to_cupy = lambda x: cupy.asarray(x)
except:
    pass
try:
    import pycuda.autoprimaryctx
    from pycuda import gpuarray
    to_gpuarray = lambda x: gpuarray.to_gpu(x)
except:
    pass

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
