"""Neural Math Module

Math Module contains a list of operators and their corresponding APIs
from the selected backend.

Consumed as follows:
>>> import neural.config
>>> neural.config.init(backend="numpy")
>>> from neural import math
>>> math.exp()  # = numpy.exp()
>>> neural.config.init(backend="cupy")
>>> math.exp()  # = cupy.exp()
>>> neural.config.init(backend="scalar")
>>> math.exp()  # = math.exp()
"""
import sys
from . import config
from .errors import NeuralBackendError

PY37 = sys.version_info >= (3, 7)

def __getattr__(attr: str):
    try:
        return getattr(config._math_engine, attr)
    except AttributeError as e:
        pass
        # raise NeuralBackendError(f"Method '{attr}' not found in {config._math_engine}") from e

def __dir__():
    return config._math_engine.__dir__()


if not PY37:
    from pep562 import Pep562
    Pep562(__name__)