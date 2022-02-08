#pylint:disable=no-member
import pytest
from neural.utils.array import (
    isarray, iscudaarray, create_empty_like,
    isiterator, get_array_module, cudaarray_to_cpu,
    iscontiguous
)
import numpy as np
import sys
from helper_funcs import to_cupy, cupy, gpuarray, to_gpuarray

ARRAYS = {
    'numpy': {
        'scalar': np.float_(0.),
        '0d': np.array(0.),
        '1d': np.array([0.]),
        '2d': np.array([[0.]]),
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
    for mod, arrays in ARRAYS.items():
        for key, arr in arrays.items():
            if arr is None:
                continue
            assert isarray(arr), f"{mod}, {key}"

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
            try:
                np.testing.assert_array_almost_equal(cpu_arr, np.asarray(ARRAYS['numpy'][key]))
            except:
                print(cpu_arr)

@pytest.mark.parametrize('conversion_f', [to_cupy, to_gpuarray])
@pytest.mark.parametrize('dtype', [np.float_, np.int_, np.float32])
def test_is_contiguous(conversion_f, dtype):
    # 0D
    a = np.array(0., dtype=dtype)
    a_C = np.ascontiguousarray(a)
    a_F = np.asfortranarray(a)
    b = conversion_f(a)
    b_C = conversion_f(a_C)
    b_F = conversion_f(a_F)
    for order in ['C', 'F']:
        for arr in [a, a_C, a_F, b, b_C, b_F]:
            assert iscontiguous(arr, order) is True

    # 1D
    a = np.arange(10).astype(dtype)
    a_C = np.ascontiguousarray(a)
    a_F = np.asfortranarray(a)
    a_dis = a[::2]
    b = conversion_f(a)
    b_C = conversion_f(a_C)
    b_F = conversion_f(a_F)
    b_dis = b[::2]
    for order in ['C', 'F']:
        for arr in [a, a_C, a_F, b, b_C, b_F]:
            assert iscontiguous(arr, order) is True
        for arr in [a_dis, b_dis]:
            assert iscontiguous(arr, order) is False

    # 2D
    a = np.random.randn(5, 10).astype(dtype)
    a_C = np.ascontiguousarray(a)
    a_F = np.asfortranarray(a)
    a_dis = a[::2, ::2]
    b = conversion_f(a)
    b_C = conversion_f(a_C)
    b_F = conversion_f(a_F)
    b_dis = b[::2, ::2]
    for arr in [a, a_C, b, b_C]:
        assert iscontiguous(arr, 'C') is True
        assert iscontiguous(arr, 'F') is False
    for arr in [a_F, b_F]:
        assert iscontiguous(arr, 'F') is True
        assert iscontiguous(arr, 'C') is False
    for arr in [a_dis, b_dis]:
        assert iscontiguous(arr, 'C') is False
        assert iscontiguous(arr, 'F') is False