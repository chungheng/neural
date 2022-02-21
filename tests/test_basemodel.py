# pylint:disable=arguments-differ
import pytest
import numpy as np
from scipy.integrate import cumtrapz
from neural.basemodel import Model
from neural.errors import NeuralModelError


class LeakyIAF(Model):
    """
    Leaky IAF neuron model.
    """

    Default_States = dict(spike=0.0, v=(-0.05, -0.070, 0.025))
    Default_Params = dict(vt=-0.025, c=1.5, vr=-0.070, r=0.2)

    def ode(self, stimulus=0.0):
        self.spike = 0.0
        self.d_v = 1.0 / self.c * (-self.v / self.r + stimulus)

    def post(self):
        self.spike = (
            1.0 if self.v > self.vt else 0.0
        )  # pylint:disable=access-member-before-definition
        self.v = self.vr if self.v > self.vt else self.v


def test_model():
    model = LeakyIAF()
    assert model.num == 1
    assert all(
        [isinstance(val, np.ndarray) and len(val) == 1 for val in model.states.values()]
    )
    assert all(
        [
            isinstance(val, np.ndarray) and len(val) == 1
            for val in model.gstates.values()
        ]
    )
    assert all(
        [isinstance(val, np.ndarray) and len(val) == 1 for val in model.params.values()]
    )
    assert model.params == LeakyIAF.Default_Params
    assert model.states == dict(spike=np.array([0]), v=np.array([-0.05]))
    assert model.Derivates == ("v",)


class DummyModel(Model):
    Default_Params = dict(a=1.0, b=0.1)
    Default_States = dict(x1=0.0, x2=(1.0, 0.0, 10.0))

    def ode(self, I_ext=0.0):
        self.d_x1 = -self.x1 * self.a
        self.d_x2 = I_ext


class DummyModel2(Model):
    Default_Params = dict(a=1.0, b=0.1)
    Default_States = dict(x1=0.0, x2=(1.0, 0.0, 10.0))

    def ode(self, not_I_ext=0.0):
        self.d_x1 = -self.x1 * self.a
        self.d_x2 = not_I_ext


class DummyModelMultiInputs(Model):
    Default_Params = dict(a=1.0, b=0.1)
    Default_States = dict(x1=0.0, x2=(1.0, 0.0, 10.0))

    def ode(self, I_ext=0.0, Input2=0.0):
        self.d_x1 = -self.x1 * self.a
        self.d_x2 = I_ext - Input2


class DummyModelMultiInputs2(Model):
    Default_Params = dict(a=1.0, b=0.1)
    Default_States = dict(x1=0.0, x2=(1.0, 0.0, 10.0))

    def ode(self, Not_I_ext=0.0, Input2=0.0):
        self.d_x1 = -self.x1 * self.a
        self.d_x2 = Not_I_ext - Input2


def res_ref(model: DummyModel, t: np.ndarray, I_ext: np.ndarray):
    return (
        model.states["x1"][:, None] * np.exp(-np.outer(model.params["a"], t)),
        cumtrapz(I_ext, axis=0, x=t, initial=0) + model.states["x2"][:, None],
    )


def res_ref_multi(
    model: DummyModelMultiInputs, t: np.ndarray, I_ext: np.ndarray, Input2: np.ndarray
):
    return (
        model.states["x1"][:, None] * np.exp(-np.outer(model.params["a"], t)),
        np.atleast_2d(np.cumsum(I_ext - Input2, axis=0).T * (t[1] - t[0]))
        + model.states["x2"][:, None],
    )


@pytest.mark.parametrize(
    "klass",
    [
        DummyModel,
        DummyModelMultiInputs,
    ],
)
def test_model_init(klass):
    model = klass(a=10.0)
    np.testing.assert_array_equal(model.params["a"], np.array([10.0]))
    np.testing.assert_array_equal(model.states["x1"], np.array([0.0]))
    np.testing.assert_array_equal(model.states["x2"], np.array([1.0]))
    assert model.bounds["x2"][0] == 0.0
    assert model.bounds["x2"][1] == 10.0
    with pytest.raises(KeyError):
        model.bounds["x1"]

    model = klass(a=10.0, num=10)
    np.testing.assert_array_equal(model.params["a"], np.full((10,), 10.0))
    np.testing.assert_array_equal(model.states["x1"], np.full((10,), 0.0))
    np.testing.assert_array_equal(model.states["x2"], np.full((10,), 1.0))
    assert model.bounds["x2"][0] == 0.0
    assert model.bounds["x2"][1] == 10.0
    with pytest.raises(KeyError):
        model.bounds["x1"]


def test_model_ode():
    model = DummyModel(a=10.0, x1=10.0)
    model.ode(I_ext=0.0)
    assert model.gstates["x1"] == -model.states["x1"] * 10.0
    assert model.gstates["x2"] == np.array([0.0])


def test_model_ode_multi_input():
    model = DummyModelMultiInputs(a=10.0, x1=10.0)
    model.ode(I_ext=0.0, Input2=2.0)
    assert model.gstates["x1"] == -model.states["x1"] * 10.0
    assert model.gstates["x2"] == np.array([-2.0])


def test_model_solve():
    dt = 1e-4
    t = np.arange(0, 0.1, dt)
    np.random.seed(0)
    I_ext = np.clip(np.random.randn(*t.shape), 0, None)
    model = DummyModel(num=1, a=1.0, x1=10.0)
    x1_ref, x2_ref = res_ref(model, t, I_ext)

    res = model.solve(t, I_ext=I_ext)
    assert res["x1"].shape == (1, len(t))
    assert res["x2"].shape == (1, len(t))
    np.testing.assert_almost_equal(x1_ref, res["x1"], decimal=4)
    np.testing.assert_almost_equal(x2_ref, res["x2"], decimal=4)

    # check I_ext dimensionality
    num = 10
    model = DummyModel(num=num)
    # 1. If I_ext is 1D, it should work with anything
    I_ext = np.random.randn(
        len(t),
    )
    model.solve(t, I_ext=I_ext)
    # 2. If I_ext is 2D, it must have the right shape
    I_ext = np.random.randn(len(t), num)
    model.solve(t, I_ext=I_ext)
    # 3. If I_ext is 2D, it must have the right shape
    I_ext = np.random.randn(len(t), num + 1)
    with pytest.raises(NeuralModelError):
        model.solve(t, I_ext=I_ext)
    # 4. If I_ext is not 1 or 2D, it will error
    I_ext = np.random.randn(len(t), 2, num + 1)
    with pytest.raises(NeuralModelError):
        model.solve(t, I_ext=I_ext)
    # 5. If provided more inputs than what ode needs, will error
    I_ext = np.random.randn(len(t), num)
    with pytest.raises(
        NeuralModelError,
        match=r"Extraneous input argument.*",
    ):
        model.solve(t, I_ext=I_ext, WrongInput1=0.0, WrongInput2=0.0)


def test_model_solve_multi():
    dt = 1e-4
    t = np.arange(0, 0.1, dt)
    np.random.seed(0)
    I_ext = np.clip(np.random.randn(*t.shape), 0, None)
    Input2 = np.clip(np.random.randn(*t.shape), 0, None)
    model = DummyModelMultiInputs(num=1, a=1.0, x1=10.0)
    x1_ref, x2_ref = res_ref_multi(model, t, I_ext, Input2)

    res = model.solve(t, I_ext=I_ext, Input2=Input2)
    assert res["x1"].shape == (1, len(t))
    assert res["x2"].shape == (1, len(t))

    model = DummyModel(num=10, a=10.0, x1=10.0)
    with pytest.raises(NeuralModelError, match=r"Extraneous input argument.*"):
        res = model.solve(t, I_ext=I_ext, Input2=Input2)

    # check I_ext dimensionality
    num = 10
    model = DummyModelMultiInputs(num=num)
    # 1. If I_ext is 1D, it should work with anything
    I_ext = np.random.randn(
        len(t),
    )
    Input2 = np.random.randn(
        len(t),
    )
    model.solve(t, I_ext=I_ext, Input2=Input2)
    # 2. If I_ext is 2D, it must have the right shape
    I_ext = np.random.randn(len(t), num)
    Input2 = np.random.randn(len(t), num)
    model.solve(t, I_ext=I_ext, Input2=Input2)
    # 3. If I_ext is 2D, it must have the right shape
    I_ext = np.random.randn(len(t), num + 1)
    Input2 = np.random.randn(len(t), num + 1)
    with pytest.raises(
        NeuralModelError,
        match=r".* must be scalar or 1D/2D array of same length as t .*",
    ):
        model.solve(t, I_ext=I_ext, Input2=Input2)
    # 4. If I_ext is not 1 or 2D, it will error
    I_ext = np.random.randn(len(t), 2, num + 1)
    Input2 = np.random.randn(len(t), 2, num + 1)
    with pytest.raises(NeuralModelError, match=r".* must be scalar or 1D/2D array.*"):
        model.solve(t, I_ext=I_ext, Input2=Input2)
