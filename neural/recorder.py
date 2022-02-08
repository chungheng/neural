# pylint:disable=no-member
"""
Utility modules for recording data from Neural models
"""
import sys
from distutils.log import warn
import numbers
from numbers import Number
import typing as tp
import numpy as np
import cupy as cp

import pycuda.gpuarray as garray
import pycuda.driver as cuda
from sympy import Integer
from . import errors as err
from . import types as tpe
from .utils import isarray, iscudaarray, get_array_module, cudaarray_to_cpu


class Recorder:
    """
    Base recorder module.

    Attributes:
    """

    def __init__(
        self,
        obj: tp.Union[tpe.Model, tpe.Operator],
        attrs: tp.Iterable[str],
        steps: int,
        rate: int = 1,
        gpu_bufsize: int = np.inf,
    ):
        """
        Arguments:
            obj: object to record rom
            attrs: attribute names to record
            steps: number of steps to record
            rate: record rate
            gpu_bufsize: size (number of steps) of gpu buffer for each gpu variable.
              If value is numpy.inf or True, then everything is recorded.
        """
        self.obj = obj
        self.total_steps = steps
        self.rate = rate
        self.steps = len(np.arange(steps)[:: self.rate])
        self.curr_step = -1  # internal counter for current step

        self.dct = {key: None for key in attrs}
        self.gpu_bufsize = self.steps if gpu_bufsize in [np.inf, True] else gpu_bufsize
        self.gpu_buf = {}
        self.spike_vars = tuple([key for key in attrs if "spike" in key.lower()])

        for key in attrs:
            src = getattr(self.obj, key)
            if isinstance(src, Number):
                shape = (self.steps,)
                gpu_shape = (self.gpu_bufsize,)
                try:
                    dtype = src.dtype
                except AttributeError:
                    if isinstance(src, int):
                        dtype = np.int_
                    elif isinstance(src, numbers.Complex):
                        dtype = np.complex_
                    else:
                        dtype = np.float_
            elif isarray(src):
                shape = (len(src), self.steps)
                gpu_shape = (len(src), self.gpu_bufsize)
                dtype = cudaarray_to_cpu(src).dtype
            else:
                raise err.NeuralRecorderError(
                    f"Attribute {key} is neither a number nor an array"
                )

            self.dct[key] = np.zeros(shape, order="F", dtype=dtype)
            if iscudaarray(src) and gpu_bufsize > 0:
                try:
                    self.gpu_buf[key] = get_array_module(src).zeros(
                        gpu_shape, dtype=src.dtype, order="F"
                    )
                except:
                    warn(
                        "Creating gpu buffer for CUDA array failed, "
                        f"source array is {self.obj}.{key}.",
                        err.NeuralRecorderWarning,
                    )

    def reset(self) -> None:
        for arr in self.dct.values():
            arr.fill(0.0)
        for arr in self.gpu_buf.values():
            arr.fill(0.0)
        self.curr_step = -1

    def update(self, index: int = None) -> None:
        """Update the content of the recorder dictionaries

        .. note::

            Storing binary variables like spike counts should
            accumulate across time bins. As such, all
            spike variables are updated every time step regardless
            of the :code:`rate` attribute. All other attributes
            are updated once every `rate` steps.
        """
        self.curr_step += 1
        step = index or self.curr_step
        d_index = int(step // self.rate)  # downsample index
        b_index = d_index % self.gpu_bufsize  # buffer index

        if d_index >= self.steps:
            raise err.NeuralRecorderError(
                "Attempting to step beyond total number of steps "
                f"({self.steps}) of recorder"
            )
        # 1. increment spike count on every step directly if spike var is not None
        for attr in self.spike_vars:
            if attr in self.gpu_buf:
                self.gpu_buf[attr][..., b_index] += getattr(self.obj, attr)
            else:
                self.dct[attr][..., d_index] += cudaarray_to_cpu(
                    getattr(self.obj, attr)
                )

        # return if not time to update
        if (step % self.rate) != 0:
            return

        for attr in self.dct.keys():
            if attr in self.spike_vars:
                continue
            # 2. increment gpu_buf if attribute is cuda array
            if attr in self.gpu_buf:
                self.gpu_buf[attr][..., b_index] = getattr(self.obj, attr)
                continue
            # 3. increment dct if attribute is not cuda array or not created in gpu_buf
            # (could be due to error)
            self.dct[attr][..., d_index] = cudaarray_to_cpu(getattr(self.obj, attr))

        # 4. dump gpu_buf to dct if gpu_buf is full or if all steps have been exhausted
        if (d_index == self.steps - 1) or (b_index == self.gpu_bufsize - 1):
            for attr, arr_g in self.gpu_buf.items():
                cudaarray_to_cpu(
                    arr_g, out=self.dct[attr][..., d_index - b_index : d_index + 1]
                )

    def __getitem__(self, key: str):
        return self.dct[key]

    def __getattr__(self, key: str):
        if key in self.dct:
            return self.dct[key]
        return super(Recorder, self).__getattribute__(key)
