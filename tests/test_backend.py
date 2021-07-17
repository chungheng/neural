import pytest
from neural.basemodel import Model
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


def test_backend():
    model = LeakyIAF()
    model.compile(backend="scalar")
    assert isinstance(model.backend, ScalarBackend)
