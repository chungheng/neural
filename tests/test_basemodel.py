import pytest
from neural.basemodel import _dict_add_, _dict_add_scalar_, _dict_iadd_, Model
from neural.backend import ScalarBackend, NumpyBackend, CUDABackend

class LeakyIAF(Model):
    """
    Leaky IAF neuron model.
    """

    Default_States = dict(spike=0, v=(-0.05, -0.070, 0.025))
    Default_Params = dict(vt=-0.025, c=1.5, vr=-0.070, r=0.2)

    def ode(self, stimulus=0.0):
        self.spike = 0.0
        self.d_v = 1.0 / self.c * (-self.v / self.r + stimulus)

    def post(self):
        if self.v > self.vt:
            self.v = self.vr
            self.spike = 1.0

@pytest.fixture
def dict_vars() -> tuple:
    a = {"a": 0, "b": 1, "c": "Hello"}
    b = {"a": 1, "b": -1, "c": " World"}
    c = 5
    ab_ref = {"a": 1, "b": 0, "c": "Hello World"}
    abc_ref = {"a": 5, "b": -4, "c": "Hello" + c * " World"}
    return (a, b, c, ab_ref, abc_ref)

def test_dict_utils(dict_vars):
    a, b, c, ab_ref, abc_ref = dict_vars
    assert _dict_add_(a, b) == ab_ref
    assert _dict_add_scalar_(a, b, c) == abc_ref
    tmp = _dict_iadd_(a, b)
    assert tmp == ab_ref
    assert id(tmp) == id(a)

def test_model():
    model = LeakyIAF()
    model.compile(backend='scalar')
    assert isinstance(model.backend, ScalarBackend)
    assert model.params == LeakyIAF.Default_Params
    assert model.states == LeakyIAF.Default_States
    