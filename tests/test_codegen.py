import pytest
import random

# pylint:disable=import-error
from neural import Model
from neural.codegen.symbolic import SympyGenerator

# pylint:enable=import-error


class FakeModel(Model):
    Default_States = dict(x=0.0, y=0.0, z=0.0)
    Default_Params = dict(a=1.0, b=2.0, c=10.0)

    def ode(self, inp1=0.0, inp2=1.0):
        self.d_x = self.a * (1 - self.x) + self.b * self.x
        self.y = self.x * self.c
        self.z = self.x * self.c
        if self.z < 1:
            self.z = 0
        else:
            self.z = 100

        self.y = (self.y > self.z) * self.y
        self.y = (self.y < self.z) * self.y
        self.y = (self.y >= self.z) * self.y
        self.y = (self.y <= self.z) * self.y

        self.y = (self.y > 2) * self.y
        self.y = (self.y < 2) * self.y
        self.y = (self.y >= 2) * self.y
        self.y = (self.y <= 2) * self.y

        self.y = (self.y > self.c) * self.y
        self.y = (self.y < self.c) * self.y
        self.y = (self.y >= self.c) * self.y
        self.y = (self.y <= self.c) * self.y


@pytest.fixture
def model():
    return FakeModel()


def test_sympy_gen(model):
    sg = SympyGenerator(model)
