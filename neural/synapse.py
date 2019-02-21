#pylint: disable=attribute-defined-outside-init
#pylint: disable=no-member
"""
Basic synapse models.
"""

from .basemodel import Model

class AMPA(Model):
    """
    AMPA Synapse
    """
    Default_States = dict(s=(0., 0., 1.))
    Default_Params = dict(ar=1e-1, ad=5e0, gmax=150.)

    def ode(self, **kwargs):
        stimulus = kwargs.pop('stimulus', 0)

        self.d_s = -self.ad*self.s

        if stimulus:
            self.s += self.ar*(1.-self.s)

class NMDA(Model):
    """
    NMDA Synapse
    """
    Default_States = dict(x=(0., 0., 1.), s=(0., 0., 1.))
    Default_Params = dict(a1=5e1, b1=4.5e0, a2=1.8e1, b2=3.4e1, gmax=300.)

    def ode(self, **kwargs):
        stimulus = kwargs.pop('stimulus', 0)

        self.d_x = -self.b1*self.x
        self.d_s = self.a2*self.x*(1.-self.s) - self.b2*self.s

        if stimulus:
            self.x += self.a1*(1.-self.x)

class GABAB(Model):
    """
    GABA_B Synapse
    """
    Default_States = dict(r=(0., 0., 1.), s=(0., 0., 1.))
    Default_Params = dict(ar=1e1, br=1e1, k3=1.8e1, k4=3.4e1, kd=.4, gmax=200., n=4)

    def ode(self, **kwargs):
        stimulus = kwargs.pop('stimulus', 0)

        self.d_r = -self.br*self.r
        self.d_s = self.k3*self.r - self.k4*self.s

        if stimulus:
            self.r += self.ar*(1.-self.r)

    def get_conductance(self):
        temp = self.s ** self.n
        half = self.kd ** self.n
        return self.gmax * temp / (temp + half)

class Exponential(Model):
    """
    Exponential Synapse
    """
    Default_States = dict(s=0.)
    Default_Params = dict(a1=1.1e1, b1=9e0, gmax=1.)

    def ode(self, **kwargs):
        stimulus = kwargs.pop('stimulus', 0)

        self.d_s = -self.b1*self.s

        if stimulus:
            self.s += self.a1

    def get_conductance(self):
        return self.s

class Alpha(Model):
    """
    Alpha Synapse
    """
    Default_States = dict(s=0., u=0.)
    Default_Params = dict(ar=1.25e1, ad=12.19, gmax=1.)

    def ode(self, **kwargs):
        stimulus = kwargs.pop('stimulus', 0)

        self.d_s = self.u
        tmp = self.ar*self.ad
        self.d_u = -(self.ar+self.ad)*self.u - tmp*self.s

        if stimulus:
            self.u += tmp
