"""Base Class of Operators
"""
import numpy as np
from ..utils.array import cudaarray_to_cpu


class Operator:
    """Operators are non-stateful operations like multiplication"""

    def __init__(self, size=None, output_size=None, dtype=np.float_):
        self.size = size
        self.dtype = dtype
        if output_size is not None:
            self.output = np.zeros(output_size, dtype=dtype)
        else:
            self.output = dtype(0.0)

    def recast(self) -> None:
        """Recast arrays to compatible formats"""
        self.output = cudaarray_to_cpu(self.output)

    def update(self, **input_args) -> None:
        pass

    def compile(self, **compiler_args) -> None:
        pass
