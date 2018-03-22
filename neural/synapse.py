#pylint: disable=attribute-defined-outside-init
"""
Basic synapse models.
"""

from basemodel import Model

class AMPA(Model):
    """
    AMPA Synapse
    """
    Default_States = dict(s=0.)
    Default_Params = dict(ar=0.5, ad=0.19, gmax=1.)

    def ode(self, spike, **kwargs):
        self.d_s = self.ar*spike*(1.-self.s) - self.ad*self.s

class NMDA(Model):
    """
    NMDA Synapse
    """
    Default_States = dict(x=0, s=0)
    Default_Params = dict(a1=0.09, b1=0.0012, a2=0.18, b2=0.034, gmax=1)

    def ode(self, spike, **kwargs):
        self.d_x = self.a1*spike*(1.-self.x) - self.b1*self.x
        self.d_s = self.a2*self.x*(1.-self.s) - self.b2*self.s

class GABAB(Model):
    """
    GABA_B Synapse
    """
    Default_States = dict(r=0, s=0)
    Default_Params = dict(ar=0.09, br=0.012, k3=0.18, k4=0.034, kd=100, gmax=1, n=4)

    def ode(self, spike, **kwargs):
        self.d_r = self.ar*spike*(1.-self.r) - self.br*self.r
        self.d_s = self.k3*self.r - self.k4*self.s

    def get_conductance(self):
        return self.gmax*self.s

class Exponential(Model):
    """
    Exponential Synapse
    """
    Default_States = dict(s=0.)
    Default_Params = dict(a1=1.1, b1=.19, gmax=1)

    def ode(self, spike, **kwargs):
        self.d_s = self.a1*spike - self.b1*self.s

class Alpha(Model):
    """
    Alpha Synapse
    """
    Default_States = dict(s=0., u=0.)
    Default_Params = dict(ar=12.5, ad=12.19, gmax=1)

    def ode(self, spike, **kwargs):
        self.d_s = self.u
        tmp = self.ar*self.ad
        self.d_u = -(self.ar+self.ad)*self.u - tmp*self.s + tmp*spike
