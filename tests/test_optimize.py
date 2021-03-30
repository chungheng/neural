import pytest
import numpy as np
import time
import pycuda.autoinit
import pycuda.gpuarray as garray
import neural.optimize
from collections import OrderedDict
from neural import Model
from neural.optimizer import differential_evolution
from neural.optimize import Optimizer


def rosen(x):
    """The Rosenbrock function"""
    r = np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, axis=0)
    return r


def rosen_batched(x):
    """The Rosenbrock function with batched parameters

    The input variable `x` is now of shape (5, N_population).
    The function works without changes because the indexing `x[1:]`
    still refers to the right variables.

    Also note that the return value `r` is now of shape (N_population,)
    """
    r = np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, axis=0)
    return r


class Rosen(Model):
    """A dummy Rosenbrock Model w/ Parameters being the optimization target"""

    Default_States = dict(r=0.0)
    Default_Params = dict(
        x=1.0,
        y=1.0,
    )

    def ode(self, stimulus=0.0):
        self.r = (1.0 - self.x) ** 2 + (self.y - self.x ** 2) ** 2


def dummy_model_input_data():
    """Input data into the Rosen Model ODE function"""
    return {"stimulus": np.arange(10)}


def rosen_model_cost(results):
    """Cost function that takes recorder data as input and energy value as output

    The results is a diction:
        - keyed by: state variable name of `Model` being optimized
        - valued by: the corresponding recorder data at step 1
    """

    # the returned data is of the shape (N_population, N_params, len(t))
    # since we have a dummy data, the rosenbrock function value is just the
    # first dimension
    return results["r"][:, 0, 0]


def test_differential_evolution():
    # non-batched rosenbrock
    bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]
    de = differential_evolution(
        rosen,
        bounds=bounds,
        verbose=False,
        batched=False,
        updating="immediate",
        seed=0,
    )
    np.testing.assert_allclose(de.x, np.ones((5,)))

    # batched rosenbrock
    de_b = differential_evolution(
        rosen_batched,
        bounds=bounds,
        verbose=False,
        batched=True,
        updating="deferred",
        seed=0,
    )
    np.testing.assert_allclose(de_b.x, np.ones((5,)))

    with pytest.warns(UserWarning, match="'polish' keyword not yet implemented"):
        de = differential_evolution(
            rosen,
            bounds=bounds,
            polish=True,
            seed=0,
        )
        np.testing.assert_allclose(de.x, np.ones((5,)))

    with pytest.warns(UserWarning, match="'batched' keyword has overridden updating"):
        de_b = differential_evolution(
            rosen_batched,
            bounds=bounds,
            verbose=False,
            batched=True,
            updating="immediate",
            seed=0,
        )
        np.testing.assert_allclose(de_b.x, np.ones((5,)))


def test_model_optimizer():
    opt = Optimizer(
        Rosen(),
        cost=rosen_model_cost,
        params=OrderedDict(x=(0, 0, 2), y=(0, 0, 2)),
        inputs=dummy_model_input_data(),
        attrs=("r",),
        dt=1,
    )
    res = opt.optimize()
    np.testing.assert_allclose(res.x, np.ones((2,)))
