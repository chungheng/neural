#pylint:disable=no-member
import pytest
from neural.utils.array import (
    isarray, iscudaarray, create_empty_like, 
    isiterator, get_array_module, cudaarray_to_cpu
)
import numpy as np
import sys
torch = None
cupy = None
gpuarray = None
try:
    import cupy
except:
    pass
try: 
    import pycuda.autoprimaryctx
    from pycuda import gpuarray
except: 
    pass
try:
    import torch
except:
    pass

ARRAYS = {
    'numpy': {
        'scalar': np.float_(0.),
        '0d': np.array(0.),
        '1d': np.array([0.]),
        '2d': np.array([[0.]]),
    },
    'torch': {
        'scalar': torch.tensor(0., device='cuda') if torch is not None else None,
        '0d': torch.from_numpy(np.array(0.)).to(device='cuda') if torch is not None else None,
        '1d': torch.from_numpy(np.array([0.])).to(device='cuda') if torch is not None else None,
        '2d': torch.from_numpy(np.array([[0.]])).to(device='cuda') if torch is not None else None,
    },
    'cupy': {
        'scalar': cupy.asarray(np.float_(0.)) if cupy is not None else None,
        '0d': cupy.asarray(np.array(0.)) if cupy is not None else None,
        '1d': cupy.asarray(np.array([0.])) if cupy is not None else None,
        '2d': cupy.asarray(np.array([[0.]])) if cupy is not None else None,
    },
    'pycuda.gpuarray': {
        'scalar': gpuarray.to_gpu(np.array(0.)) if gpuarray is not None else None,
        '0d': gpuarray.to_gpu(np.array(0.)) if gpuarray is not None else None,
        '1d': gpuarray.to_gpu(np.array([0.])) if gpuarray is not None else None,
        '2d': gpuarray.to_gpu(np.array([[0.]])) if gpuarray is not None else None,
    },
}

def test_isarray():
    for key, arr in ARRAYS['numpy'].items():
        assert isarray(arr)

def test_iscudaarray():
    for mod, arrays in ARRAYS.items():
        for key, arr in arrays.items():
            if arr is None:
                continue
            if mod == 'numpy':
                assert not iscudaarray(arr), f"{mod}, {key}"
            else:
                assert iscudaarray(arr), f"{mod}, {key}"

def test_isiterator():
    for mod, arrays in ARRAYS.items():
        for key, arr in arrays.items():
            if arr is None:
                continue
            assert not isiterator(arr), f"{mod}, {key}"
    
    assert isiterator((x for x in range(10)))

def test_get_array_module():
    for mod, arrays in ARRAYS.items():
        for key, arr in arrays.items():
            if arr is None:
                continue
            assert sys.modules[mod] == get_array_module(arr), \
                f"{mod}, {key}"

def test_cudaarray_to_cpu():
    for mod, arrays in ARRAYS.items():
        if mod =='numpy':
            continue
        for key, arr in arrays.items():
            if arr is None:
                continue
            cpu_arr = cudaarray_to_cpu(arr)
            np.testing.assert_array_almost_equal(cpu_arr, np.asarray(ARRAYS['numpy'][key]))