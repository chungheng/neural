import random

import numpy as np
from warnings import warn
import pycuda.driver as drv
import pycuda.gpuarray as garray
from pycuda.elementwise import ElementwiseKernel

from neural.basemodel import Model
from neural.network.operator import Operator

class Mixture(Operator):
    def __init__(self, size, aff1=None, aff2=None, dtype=np.float64, **kwargs):
        size = kwargs.pop('size', size)
        backend = kwargs.pop('backend', 'cuda')
        super().__init__(size=size, dtype=dtype, backend=backend)
        if aff1.dtype != dtype:
            aff1 = aff1.astype(dtype)
        if aff2.dtype != dtype:
            aff2 = aff2.astype(dtype)
        self.aff1 = garray.to_gpu(aff1)
        self.aff2 = garray.to_gpu(aff2)

    def update(self, input1, input2):
        self._update(input1, input2, self.output, self.aff1, self.aff2)

    def compile(self, **kwargs):
        if self._backend == 'cuda':
            self._update = ElementwiseKernel(
                "{0} in1, {0} in2, {0} *out, {0} *aff1, {0} *aff2".format(self.dtype),
                "out[i] = in1*aff1[i] + in2*aff2[i];",
                "Mixture")
        else:
            def f(in1, in2, aff1, aff2):
                self.output = in1*aff1 + in2*aff2
            self._update = lambda in1, in2, o, aff1, aff2: f(in1, in2, aff1, aff2)

class MixtureN(Operator):
    def __init__(self, size, aff, dtype=np.float64, **kwargs):
        backend = kwargs.pop('backend', 'cuda')
        super().__init__(size=size, dtype=dtype, backend=backend)
        self.aff_g = []
        for _aff in aff:
            self.aff_g.append(garray.to_gpu(_aff.astype(dtype)))

    def update(self, **inputs_dict):
        inputs = [None]*len(inputs_dict)
        for n, inp in inputs_dict.items():
            inputs[int(n)] = inp
        self._update(*inputs, self.output, *self.aff_g)
        if self.backend != 'cuda':
            if isinstance(self.output, garray.GPUArray):
                self.output = self.output.get()

    def compile(self, **kwargs):
        N_pure = len(self.aff_g)
        if self._backend != 'cuda':
            warn('MixtureN requested compilation of CPU-mode, output post-processed to be CPU compatible')
        inputs = ", ".join(["{} in{}".format(self.dtype, n) for n in range(N_pure)])
        affs = ", ".join(["{} *aff{}".format(self.dtype, n) for n in range(N_pure)])
        self._update = ElementwiseKernel(
            inputs + ", {0} *out, ".format(self.dtype) +  affs,
            "out[i] = " + " + ".join(["in{0}*aff{0}[i]".format(n) for n in range(N_pure)]) + ";",
            "MixtureN")

class NoisyIAF(Model):
    """
    Integrate-and-Fire Neuron Model with Noise.
    """
    Default_States = dict(
        v=(0, 0, 10),
        spike=0,
        count=0.)
    Default_Params = dict(vt=0.025, c=5., bias=0.01, s=1, sigma=0.001)

    def ode(self, stimulus=0.):

        self.spike = self.v > self.vt
        self.v = (self.spike < 1) * self.v
        self.d_v = 1./self.c*(self.s*stimulus+self.bias) + np.random.normal(0,  self.sigma)
        

    def post(self):
        if self.v > self.vt:
            self.count += self.spike

class OTP(Model):
    Default_States = dict(
        v=0.,
        I=0.,
        uh=(0., 0., 50000.),
        duh=0.,
        x1=(0., 0., 1.),
        x2=(0., 0., 1.),
        x3=(0., 0., 1000.))
    Default_Params = dict(
        bf=1.0,
        gamma=0.13827484362015477,
        d1=10.06577381490841,
        c1=0.02159722808408414,
        a2=199.57381809612792,
        b2=51.886883149283406,
        a3=2.443964363230107,
        b3=0.9236173421313049,
        k23=9593.91481121941,
        CCR=0.06590587362782163,
        ICR=91.15901333340182,
        L=0.8,
        W=45.)
    
    def ode(self, stimulus=0.):
        self.d_uh = self.duh
        self.d_duh = -2*self.W*self.L*self.duh + self.W*self.W*(stimulus-self.uh)
        self.v = self.uh + self.gamma*self.duh
        self.v = (self.v > 0) * self.v

        self.d_x1 = self.c1*self.bf*self.v*(1.-self.x1) - self.d1*self.x1
        f = np.cbrt(self.x2*self.x2) * np.cbrt(self.x3*self.x3)
        self.d_x2 = self.a2*self.x1*(1.-self.x2) - self.b2*self.x2 - self.k23*f
        self.d_x3 = self.a3*self.x2 - self.b3*self.x3

        self.I = self.ICR * self.x2 / (self.x2 + self.CCR)
        

class PoissonCSN(Model):
    Default_States = dict(x=0.,r=0., spike=0., cx=0.)
    Default_Params = dict(
        x6= 2.79621009e-09,
        x5=-9.55636291e-07,
        x4= 1.25880567e-04,
        x3=-7.79496241e-03,
        x2= 1.94672932e-01,
        x1= 3.44246777,
        x0= 5.11085315)

    def ode(self, I=0.):

        self.x = 0.
        Ip = 1.
        self.d_x = Ip*self.x0
        Ip = Ip*I
        self.d_x += Ip*self.x1
        Ip = Ip*I
        self.d_x += Ip*self.x2
        Ip = Ip*I
        self.d_x += Ip*self.x3
        Ip = Ip*I
        self.d_x += Ip*self.x4
        Ip = Ip*I
        self.d_x += Ip*self.x5
        Ip = Ip*I
        self.d_x += Ip*self.x6
        
        self.r = random.uniform(0., 1.0)
                
    def post(self):
        self.spike = (self.r < self.x)
        self.cx += self.x
    
class OSNiLNSynapse(Model):
    """
    OSN to pre-iLN Synapse
    """
    Default_States = dict(
        x = (0., 0., 1e9),
        y = 0.,
    )
    Default_Params = dict(
        a = 1.49055402e-02,
        b = 1.86235685e+02,
        c = 5.55168635e-04,
        s = 800
    )
    def ode(self, stimulus=0.):
        self.d_x = self.s*(stimulus-self.c)*(self.b+self.x) - self.s*self.a*self.x
        self.y = self.x*self.x

class OSNAxonTerminalRate(Model):
    """
    OSN Axon Terminal Model
    
    The input to the model is assumed to be OSN spike rate.
    """
    Default_States = dict(
        x = (0., 0., 1e9),
        u = (0., 0., 1e9)
    )
    Default_Params = dict(
        a = 1.49055402e-02,
        b = 1.86235685e+02,
        c = 5.55168635e-04,
        s = 800,
        d = 150
    )
    def ode(self, stimulus=0., f=1.):
        self.d_x = self.s*(stimulus-self.c)*(self.b+self.x) - self.s*self.a*self.x
#         self.d_u = 100*self.x - 100*self.u * (f + self.d)
        self.u = self.x / (f + self.d)
        

class PostLN(Model):
    """
    Post-Synaptic Local Neuron Model.
    
    
    """
    Default_States = dict(
        v = 0.,
        i = 0.,
        x = 0.,
        y = 0.,
        z = 0.
    )
    Default_Params = dict(
        s = 100.,
        p = 1.,
        R = 3.,
        C = 0.01,#0.03
        L = 0.5,#1.,
        tau = 4./300/1.38*1e4,
        RC = 470*1e-4,
        C1 = 470*1e-2,
        a = 1e-1
    )
    def ode(self, stimulus=0.):
        self.d_v = ( self.i/self.C  + (self.s*stimulus - self.v)/(self.R * self.C))
        self.d_i = (self.s*stimulus - self.v)/self.L
        self.d_x = -self.x/self.RC + self.y/self.C1
        self.y = 1e-1*(np.exp(self.tau*(self.p*self.i-self.x))-1)
        self.z = self.a*self.y
        
class KCDend(Model):
    """
    Tentative KC Dendritic Processing Model
    """
    Default_States = dict(
        x=(0., 0., 10000.),
        y=(0., 0., 10000.),
        z=(0., 0., 10000.),
        dz=(0., 0., 10000.)
    )
    Default_Params = dict(
        b=0.1,
        th=2.)

    def ode(self, stimulus=0., f=1.):
        self.d_z = -1000*self.z + 1000*f
#         self.x = stimulus / (self.b + 3*self.z)
        _rate = 10
        self.d_x = _rate*stimulus - _rate*(self.b + 3*self.z)*self.x
        
class ReLU(Model):
    """
    Rectified Linear Unit Model
    """
    Default_States = dict(x=0.)
    Default_Params = dict(th=0.)
    def ode(self, stimulus=0.):

        self.x = stimulus * (stimulus >= self.th)
        
class NoisyConnorStevens(Model):
    """
    Connor-Stevens Model
    """
    Time_Scale = 1e3 # s to ms
    Default_States = dict(
        v=(-60, -80, 80),
        n=(0., 0., 1.),
        m=(0., 0., 1.),
        h=(1., 0., 1.),
        a=(1., 0., 1.),
        b=(1., 0., 1.),
        spike=0, v1=-60., v2=-60., refactory=0.)
    Default_Params = dict(ms=-5.3, ns=-4.3, hs=-12., \
        gNa=120., gK=20., gL=0.3, ga=47.7, \
        ENa=55., EK=-72., EL=-17., Ea=-75., \
        sigma=2.05, refperiod=1.5)

    def ode(self, stimulus=0.):

        alpha = np.exp(-(self.v+50.+self.ns)/10.)-1.
        if abs(alpha) <= 1e-7:
            alpha = 0.1
        else:
            alpha = -0.01*(self.v+50.+self.ns)/alpha
        beta = .125*np.exp(-(self.v+60.+self.ns)/80.)
        n_inf = alpha/(alpha+beta)
        tau_n = 2./(3.8*(alpha+beta))

        alpha = np.exp(-(self.v+35.+self.ms)/10.)-1.
        if abs(alpha) <= 1e-7:
            alpha = 1.
        else:
            alpha = -.1*(self.v+35.+self.ms)/alpha
        beta = 4.*np.exp(-(self.v+60.+self.ms)/18.)
        m_inf = alpha/(alpha+beta)
        tau_m = 1./(3.8*(alpha+beta))

        alpha = .07*np.exp(-(self.v+60.+self.hs)/20.)
        beta = 1./(1.+np.exp(-(self.v+30.+self.hs)/10.))
        h_inf = alpha/(alpha+beta)
        tau_h = 1./(3.8*(alpha+beta))

        a_inf = np.cbrt(.0761*np.exp((self.v+94.22)/31.84)/(1.+np.exp((self.v+1.17)/28.93)))
        tau_a = .3632+1.158/(1.+np.exp((self.v+55.96)/20.12))
        b_inf = np.power(1/(1+np.exp((self.v+53.3)/14.54)), 4.)
        tau_b = 1.24+2.678/(1+np.exp((self.v+50)/16.027))

        i_na = self.gNa * np.power(self.m, 3) * self.h * (self.v - self.ENa)
        i_k = self.gK * np.power(self.n, 4) * (self.v - self.EK)
        i_l = self.gL * (self.v - self.EL)
        i_a = self.ga * np.power(self.a, 3) * self.b * (self.v - self.Ea)

        self.d_v = stimulus - i_na - i_k - i_l - i_a
        self.d_n = (n_inf-self.n)/tau_n + random.gauss(0., self.sigma)
        self.d_m = (m_inf-self.m)/tau_m + random.gauss(0., self.sigma)
        self.d_h = (h_inf-self.h)/tau_h + random.gauss(0., self.sigma)
        self.d_a = (a_inf-self.a)/tau_a + random.gauss(0., self.sigma)
        self.d_b = (b_inf-self.b)/tau_b + random.gauss(0., self.sigma)

        self.d_refactory = (self.refactory < 0)

    def post(self):
        self.spike = (self.v1 < self.v2) * (self.v < self.v2) * (self.v > -30.)
        self.v1 = self.v2
        self.v2 = self.v
        self.spike = (self.spike > 0.) * (self.refactory >= 0)
        self.refactory -= (self.spike > 0.) * self.refperiod
