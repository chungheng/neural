import re
import inspect
import sys
from types import ModuleType
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
        return mod # torch or pycuda
    if isinstance(arr, (np.ndarray, np.generic)):
        return np # numpy    
    if any(['cupy' in s for s in re.findall(r"\<class *\'(.*)\'\>", str(arr.__class__))]):
        return sys.modules['cupy']
    raise err.NeuralUtilityError(f"Cannot get array module of array: {arr}")

def create_empty_like(arr: npt.ArrayLike) -> npt.ArrayLike:
    if not isarray(arr):
        raise err.NeuralUtilityError("Argument is not array")
    if not iscudaarray(arr):
        return np.empty_like(arr)
    return get_array_module(arr).empty_like(arr)

def cudaarray_to_cpu(arr:npt.ArrayLike, out: npt.ArrayLike = None) -> npt.ArrayLike:
    if not iscudaarray(arr):
        return None
    if hasattr(arr, 'get'):
        try:
            return arr.get(out=out) # cupy
        except TypeError:
            return arr.get(ary=out) # pycuda
        except Exception as e:
            raise err.NeuralUtilityError(
                f"Cannot convert cuda array to numpy array"
            ) from e
    if hasattr(arr, 'detach'):
        try:
            return arr.detach().cpu().numpy() # torch
        except Exception as e:
            raise err.NeuralUtilityError(
                f"Cannot convert cuda array to numpy array"
            ) from e
    raise err.NeuralUtilityError(
        f"cuda array type not supported. does not have .get() or .detach() methods"
    )