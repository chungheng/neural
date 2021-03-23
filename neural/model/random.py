"""
Basic random point process.
"""
import random
import numpy as np

from .basemodel import Model


class PoissonProcess(Model):
    """
    Poisson point process for generating spike.
    """

    Default_Inters = dict(spike=0, r=0.0)
    Default_Params = dict(x=0.0)

    def ode(self, **kwargs):
        stimulus = kwargs.pop("stimulus", 0)

        self.d_x = stimulus
        self.r = random.uniform(0.0, 1.0)

    def post(self):
        self.spike = self.x > self.r
