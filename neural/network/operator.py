# pylint:disable=arguments-differ
from functools import reduce
import operator
import numpy as np
from .baseoperator import Operator


class Add(Operator):
    def update(self, **input):
        self.output = reduce(operator.add, input.values())


class Square(Operator):
    def update(self, input=0.0):
        self.output = input**2


class Sqrt(Operator):
    def update(self, input=0.0):
        self.output = input**0.5


class Sum(Operator):
    def update(self, input=0.0):
        self.output = np.sum(input)


class BlockSum(Operator):
    def __init__(self, **kwargs):
        block_size = kwargs.pop("block_size", 1)
        self.block_size = block_size
        kwargs["output_size"] = int(kwargs["size"] // block_size)
        super().__init__(**kwargs)

    def update(self, input=0.0):
        _input = input.reshape(-1, self.block_size)
        self.output = np.sum(_input)


class Mean(Operator):
    def update(self, input=0.0):
        self.output = np.mean(input)


class BlockMean(Operator):
    def __init__(self, block_size=1, **kwargs):
        self.block_size = block_size
        kwargs["output_size"] = int(kwargs["size"] // block_size)
        super().__init__(**kwargs)

    def update(self, input=0.0):
        _input = input.reshape(-1, self.block_size)
        self.output = np.mean(_input, axis=1)


class Repeat(Operator):
    def __init__(self, rep_size, **kwargs):
        kwargs["output_size"] = rep_size * kwargs["size"]
        super().__init__(**kwargs)
        self.rep_size = rep_size

    def update(self, input=0.0):
        self.output = np.repeat(input, self.rep_size)


class Dot(Operator):
    def __init__(self, multiplier, batch_size=1, **kwargs):
        kwargs["output_size"] = multiplier.shape[0] * batch_size
        super().__init__(**kwargs)
        if isinstance(multiplier, np.ndarray):
            multiplier = multiplier.astype(self.dtype)
            multiplier = np.asfortranarray(multiplier)
            self.multiplier = multiplier

        self.batch_size = batch_size
        self._output = self.output.reshape(-1, batch_size, order="F")

    def update(self, input=0.0):

        _input = input.reshape(-1, self.batch_size, order="F")
        self.output = xp.dot(self.multiplier, _input)
