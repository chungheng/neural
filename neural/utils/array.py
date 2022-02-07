from numbers import Number
import re
import inspect
import sys
from types import ModuleType
from typing import Literal
import numpy as np
import numpy.typing as npt
from .. import errors as err

def iscudaarray(arr) -> bool:
    if hasattr(arr, "__cuda_array_interface__"):
        return True
    return False

def isarray(arr) -> bool:
    if hasattr(arr, "__array_interface__") or iscudaarray(arr):
        return True
    return False

def isiterator(arr) -> bool:
    if hasattr(arr, "__next__") and hasattr(arr, "__iter__"):
        return True
    return False

def get_array_module(arr: npt.ArrayLike) -> ModuleType:
    if not isarray(arr):
        return None
    if (mod := inspect.getmodule(arr)) is not None:
        return mod # pycuda.gpuarray
    if isinstance(arr, (np.ndarray, np.generic)):
        return np # numpy
    if any([
        'cupy' in s
        for s
        in re.findall(r"\<class *\'(.*)\'\>", str(arr.__class__))
    ]):
        return sys.modules['cupy']
    raise err.NeuralUtilityError(f"Cannot get array module of array: {arr}")

def create_empty_like(arr: npt.ArrayLike) -> npt.ArrayLike:
    if not isarray(arr):
        raise err.NeuralUtilityError("Argument is not array")
    if not iscudaarray(arr):
        return np.empty_like(arr)
    return get_array_module(arr).empty_like(arr)

def cudaarray_to_cpu(arr:npt.ArrayLike, out: npt.ArrayLike = None) -> npt.ArrayLike:
    if not isarray(arr):
        raise err.NeuralUtilityError("Cannot convert non-array to numpy array")
    if not iscudaarray(arr):
        return arr
    if hasattr(arr, 'get'):
        try:
            return arr.get(out=out) # cupy
        except TypeError:
            return arr.get(ary=out) # pycuda
        except Exception as e:
            raise err.NeuralUtilityError(
                "Cannot convert cuda array to numpy array"
            ) from e
    raise err.NeuralUtilityError(
        "cuda array type not supported. does not have .get() or .detach() methods"
    )

def iscontiguous(arr: npt.ArrayLike, order: Literal['C', 'F']="C") -> bool:
    if not isarray(arr):
        raise err.NeuralUtilityError("Input must be array")
    if hasattr(arr, 'flags'): # cupy, pycuda, numpy
        if order == 'C':
            return arr.flags.c_contiguous
        if order == 'F':
            return arr.flags.f_contiguous
        raise err.NeuralUtilityError("order must be 'C' or 'F'")
    raise err.NeuralUtilityError(f"Array of type '{type(arr)}' not understood.")