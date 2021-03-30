"""Parameter Optimization of Model and Network

This module provides helper functions for optimizing parameter values of a
given model or network via global optimization like Differential Evolution
and SHGO. Because of the complex nature of biological neural networks,
optimization via backpropagation is difficult. However, it is possible to
optimize for parameters globally using non-convex methods.
"""
import sys
from inspect import signature
import typing as tp
from collections import OrderedDict
import warnings
import numpy as np
from pycuda import gpuarray
from .optimizer import differential_evolution
from .basemodel import Model
from .network.operator import Repeat
from .network import Network
from . import errors as err

PY37 = (sys.version_info.major * 10 + sys.version_info.minor) >= 37


class Optimizer:
    def __new__(self, target, cost, params, inputs, attrs=None, dt=1.0, **kwargs):
        if isinstance(target, Network):
            raise NotImplementedError("Network Optimization is not available.")
        if isinstance(target, Model):
            constructor = target.__class__
            return ModelOptimizer(
                constructor, cost, params, inputs, attrs, dt, **kwargs
            )
        if issubclass(target, Model):
            constructor = target
            return ModelOptimizer(
                constructor, cost, params, inputs, attrs, dt, **kwargs
            )
        raise TypeError(f"target of type {type(target)} not understood.")


class ModelOptimizer:
    """Optimizer for Neural.Model

    Arguments:
        constructor:
        cost: cost function that takes in target model recorder as input
        params:
        inputs:
        attrs: model attributes to compute cost function from
        batchsize:
        maxiter:
        dtype:
        verbose

    Attributes:
        _Nchannel: Number of parameters being optimized
    """

    def __init__(self, constructor, cost, params, inputs, attrs, dt, **kwargs):
        self.dt = dt
        self.constructor = constructor
        self.cost = cost
        self.batchsize = kwargs.pop("batchsize", 100)
        self.maxiter = kwargs.pop("maxiter", 100)
        self.verbose = kwargs.pop("verbose", True)
        self.dtype = kwargs.pop("dtype", np.float64)
        self._Nchannel, self.inputs = self._inputs_validator(constructor, inputs)
        self.params = self._params_validator(constructor, params)
        self.bounds = [val[1:] for val in self.params.values()]

        num_population_size = self.batchsize * len(self.params)
        self.num = num_population_size * self._Nchannel
        self._nn = self._create_network({})

    def optimize(self):
        return differential_evolution(
            self.objective_func,
            self.bounds,
            maxiter=self.maxiter,
            popsize=self.batchsize,
            batched=True,
            verbose=self.verbose,
        )

    def objective_func(self, x: np.ndarray):
        if x.shape != (len(self.params), len(self.params) * self.batchsize):
            x = x.T
        assert x.shape == (len(self.params), len(self.params) * self.batchsize)

        new_params = {
            key: np.tile(np.squeeze(x[n]), self._Nchannel)
            for n, key in enumerate(self.params.keys())
        }
        self._nn.containers["Target"].obj.params.update(**new_params)
        self._nn.compile(dtype=self.dtype)
        self._nn.run(self.dt, verbose=False)

        # reshape data into (Nparams, Nchannel, Nt) shape
        # where Nchannel is the number of components for each parameter set
        # e.g. Nchannel can refer to number of receptor channels in an Antennal
        # Lobe circuit
        target_result = {
            k: np.swapaxes(val.reshape((self._Nchannel, -1, val.shape[1])), 0, 1)
            for k, val in self._nn.containers["Target"].recorder.dct.items()
        }
        return self.cost(target_result)

    def _create_network(self, params):
        nn = Network()
        inp_symbols = dict()
        for key, val in self.inputs.items():
            # create input nodes and fill data
            inp = nn.input(num=self._Nchannel, name=key)
            inp(val)
            # repeat input nodes by batchsize
            rep = nn.add(
                Repeat,
                self._Nchannel,
                rep_size=self.batchsize * len(self.params),
                name=f"Repeat-{key}",
            )
            rep(input=inp)
            inp_symbols[key] = rep.output

        target = nn.add(self.constructor, self.num, name="Target", **params)
        target(**inp_symbols)
        target.record(*list(target.obj.states.keys()))
        nn.compile(dtype=self.dtype)
        return nn

    def _inputs_validator(self, constructor: Model, inputs: dict):
        args = signature(constructor.ode).parameters

        # check that input names are valid
        for k in inputs:
            if k not in args:
                raise ValueError(f"inputs '{k}' not valid for model {constructor}")

        # conform input shapes to have time dimension in first dimension,
        # also check time dimension is the same across all inputs
        # it also chekcs if the number of channels are the same and broadcasts
        # the 1D arrays to (Ntime, Nchannel) shape
        Nt = []
        Nc = dict()
        for key, val in inputs.items():
            if isinstance(val, np.ndarray):
                val = np.ascontiguousarray(np.squeeze(val))
            elif isinstance(val, gpuarray.GPUArray):
                val = val.squeeze()
            else:
                raise TypeError(f"Input '{key}' is neither NumPy nor PyCuda Array.")

            if val.ndim > 2:
                raise ValueError(f"Input '{key}' is more than 2D.")

            Nt.append(val.shape[0])
            Nc[key] = 1 if val.ndim == 1 else val.shape[1]
            inputs[key] = val

        if len(set(Nc.values())) > 2:
            raise ValueError("Channel number inconsistent.")
        if len(set(Nt)) > 1:
            raise ValueError("Input time dimension not consistent.")

        # if the number of channels is more than 1, then broadcast the
        # inputs with channel = 1 to the required number
        Nchannel = np.max(list(set(Nc.values())))
        if Nchannel > 1:
            for key, val in Nc.items():
                if val == 1:
                    if isinstance(val, gpuarray.GPUArray):
                        val = val.get()
                    val = np.ascontiguousarray(
                        np.repeat(val[:, None], Nchannel, axis=1)
                    )
                    inputs[key] = val

        # convert everything to GPUArray
        for key, val in inputs.items():
            if isinstance(val, np.ndarray):
                val = gpuarray.to_gpu(np.ascontiguousarray(val))
                inputs[key] = val

        return Nchannel, inputs

    def _params_validator(
        self, constructor: Model, params: tp.Union[dict, OrderedDict]
    ):
        """Validate Parameters and Conform to Model Specification"""
        if not isinstance(params, dict):
            raise TypeError(
                f"Params of type {type(params)} not understood, must be dict-like."
            )

        if not isinstance(params, OrderedDict) and not PY37:
            warnings.warn(
                "dictionary params can lose insertion order before Python 3.7"
                " . Params will be converted OrderedDict to ensure ordering as "
                " much as possible, but you should do this locally when passing"
                " in params value."
            )
            params = OrderedDict(params)

        for key, val in params.items():
            if len(val) != 3:
                raise ValueError(
                    f" Parameter '{key}' is not specified as a 3-tuple of"
                    f" (initial, lower bound, upper bound)"
                )
            init, lbound, ubound = val
            if ubound < lbound:
                raise ValueError(
                    f"Upper Bound {ubound} cannot be smaller than Lower "
                    f" Bound {lbound}"
                )
            if ubound == lbound:
                raise ValueError(
                    f"Upper Bound {ubound} cannot be the same as Lower "
                    f" Bound {lbound}"
                )

            # make sure initial guess is within bound
            init = np.clip(init, lbound, ubound)

            if key not in constructor.Default_Params:
                raise AttributeError(
                    f"Parameter '{key}' not found in target '{target}'"
                ) from e
            # else:
            #     default_val = constructor.Default_Params[key]
            #     if np.isscalar(default_val):
            #         continue
            #     if len(default_val) == 3:
            #         df_init,df_lbound,df_ubound = default_val

            #         if lbound > df_ubound or ubound < df_lbound:
            #             raise ValueError(
            #                 f"Perimissible range of parameter '{key}' is not "
            #                 f" Contained in the range specified in model"
            #                 f" constructor, cannot optimize"
            #             )
            #         if lbound < df_lbound:
            #             lbound = df_lbound
            #         if ubound > df_ubound:
            #             ubound = df_ubound
            params[key] = (init, lbound, ubound)
        return params
