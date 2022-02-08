# pylint: disable=attribute-defined-outside-init
# pylint: disable=access-member-before-definition
# pylint:disable=arguments-differ
# pylint:disable=abstract-method
# pylint:disable=invalid-name
"""
Basic neuron models.
"""
import math
from ..basemodel import Model


class IAF(Model):
    """
    Integrate-and-Fire neuron model.
    """

    Default_States = dict(spike=0, v=0)
    Default_Params = dict(vt=0.025, c=5.0, bias=0.01)

    def ode(self, stimulus=0.0):
        self.spike = 0.0
        self.d_v = 1.0 / self.c * (stimulus + self.bias)

    def post(self):
        self.spike = 1.0 if self.v > self.vt else 0.0
        self.v = 0.0 if self.v > self.vt else self.v


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
        self.spike = 1.0 if self.v > self.vt else 0.0
        self.v = self.vr if self.v > self.vt else self.v


class Rinzel(Model):
    """
    Rinzel neuron model.
    """

    Time_Scale = 1e3  # s to ms
    Default_States = dict(v=(-60, -80, 30), w=(0.0, 0.0, 1.0))
    Default_Params = dict(
        C=0.75,
        g_Na=120.0,
        g_K=36.0,
        g_L=0.3,
        E_Na=50.0,
        E_K=-77.0,
        E_L=-54.387,
        s=1.27135220916422,
    )

    def ode(self, stimulus=0.0):

        alpha = math.exp(-(self.v + 55.0) / 10.0) - 1.0
        beta = 0.125 * math.exp(-(self.v + 65.0) / 80.0)
        alpha = 0.1 if abs(alpha) <= 1e-7 else -0.01 * (self.v + 55.0) / alpha
        n_infty = alpha / (alpha + beta)

        alpha = math.exp(-(self.v + 40.0) / 10.0) - 1.0
        beta = 4.0 * math.exp(-(self.v + 65.0) / 18.0)
        alpha = 1.0 if abs(alpha) <= 1e-7 else -0.1 * (self.v + 40.0) / alpha
        m_infty = alpha / (alpha + beta)

        alpha = 0.07 * math.exp(-(self.v + 65.0) / 20.0)
        beta = 1.0 / (math.exp(-(self.v + 35.0) / 10.0) + 1.0)
        h_infty = alpha / (alpha + beta)

        w_infty = self.s / (1.0 + self.s**2) * (n_infty + self.s * (1.0 - h_infty))
        tau_w = 1.0 + 5.0 * math.exp(-(self.v + 55.0) * (self.v + 55.0) / 55.0 * 55.0)

        self.d_w = 3.0 * w_infty / tau_w - 3.0 / tau_w * self.w

        i_na = self.g_Na * (m_infty**3) * (1.0 - self.w) * (self.v - self.E_Na)
        i_k = self.g_K * ((self.w / self.s) ** 4) * (self.v - self.E_K)
        i_l = self.g_L * (self.v - self.E_L)

        self.d_v = 1.0 / self.C * (stimulus - i_na - i_k - i_l)


class Wilson(Model):
    """
    Wilson neuron model.

    [1] Wilson, "Spikes, decisions, and actions," 1998.
    """

    Time_Scale = 1e3  # s to ms
    Default_States = dict(r=0.088, v=-70.0)
    Default_Params = dict(C=1.2, E_K=-92.0, g_K=26.0, E_Na=55.0)

    def ode(self, stimulus=0.0):

        r_infty = 0.0135 * self.v + 1.03
        self.d_r = (r_infty - self.r) / 1.9

        i_na = (17.81 + 0.4771 * self.v + 0.003263 * self.v**2) * (self.v - self.E_Na)
        i_k = self.g_K * self.r * (self.v - self.E_K)
        self.d_v = 1.0 / self.C * (stimulus - i_na - i_k)


# class MorrisLecar(Model):
#     """
#     Morris-Lecar neuron model.
#     """

#     Default_States = dict(r=0, s=0)
#     Default_Params = dict(ar=0.09, br=0.012, k3=0.18, k4=0.034, kd=100, gmax=1, n=4)

#     def ode(self, **kwargs):
#         pass


class ConnorStevens(Model):
    """
    Connor-Stevens neuron model.
    """

    Time_Scale = 1e3  # s to ms
    Default_States = dict(
        v=(-60, -80, 50),
        n=(0.0, 0.0, 1.0),
        m=(0.0, 0.0, 1.0),
        h=(1.0, 0.0, 1.0),
        a=(1.0, 0.0, 1.0),
        b=(1.0, 0.0, 1.0),
    )
    Default_Params = dict(
        ms=-5.3,
        ns=-4.3,
        hs=-12.0,
        g_Na=120.0,
        g_K=20.0,
        g_L=0.3,
        g_a=47.7,
        E_Na=55.0,
        E_K=-72.0,
        E_L=-17.0,
        E_a=-75.0,
    )

    def ode(self, stimulus=0.0):

        alpha = math.exp(-(self.v + 50.0 + self.ns) / 10.0) - 1.0
        alpha = (
            0.1 if abs(alpha) <= 1e-7 else -0.01 * (self.v + 50.0 + self.ns) / alpha
        )
        beta = 0.125 * math.exp(-(self.v + 60.0 + self.ns) / 80.0)
        n_inf = alpha / (alpha + beta)
        tau_n = 2.0 / (3.8 * (alpha + beta))

        alpha = math.exp(-(self.v + 35.0 + self.ms) / 10.0) - 1.0
        alpha = (
            1.0 if abs(alpha) <= 1e-7 else -0.1 * (self.v + 35.0 + self.ms) / alpha
        )
        beta = 4.0 * math.exp(-(self.v + 60.0 + self.ms) / 18.0)
        m_inf = alpha / (alpha + beta)
        tau_m = 1.0 / (3.8 * (alpha + beta))

        alpha = 0.07 * math.exp(-(self.v + 60.0 + self.hs) / 20.0)
        beta = 1.0 / (1.0 + math.exp(-(self.v + 30.0 + self.hs) / 10.0))
        h_inf = alpha / (alpha + beta)
        tau_h = 1 / (3.8 * (alpha + beta))

        a_inf = (
            0.0761
            * math.exp((self.v + 94.22) / 31.84)
            / (1.0 + math.exp((self.v + 1.17) / 28.93))
        ) ** (1./3.)

        tau_a = 0.3632 + 1.158 / (1.0 + math.exp((self.v + 55.96) / 20.12))
        b_inf = pow(1 / (1 + math.exp((self.v + 53.3) / 14.54)), 4.0)
        tau_b = 1.24 + 2.678 / (1 + math.exp((self.v + 50) / 16.027))

        i_na = self.g_Na * pow(self.m, 3) * self.h * (self.v - self.E_Na)
        i_k = self.g_K * pow(self.n, 4) * (self.v - self.E_K)
        i_l = self.g_L * (self.v - self.E_L)
        i_a = self.g_a * pow(self.a, 3) * self.b * (self.v - self.E_a)

        self.d_v = stimulus - i_na - i_k - i_l - i_a
        self.d_n = (n_inf - self.n) / tau_n
        self.d_m = (m_inf - self.m) / tau_m
        self.d_h = (h_inf - self.h) / tau_h
        self.d_a = (a_inf - self.a) / tau_a
        self.d_b = (b_inf - self.b) / tau_b


class HodgkinHuxley(Model):
    """
    Hodgkin-Huxley neuron model.
    """

    Time_Scale = 1e3  # s to ms
    Default_States = dict(
        v=(-60, -80, 30), n=(0.0, 0.0, 1.0), m=(0.0, 0.0, 1.0), h=(1.0, 0.0, 1.0)
    )
    Default_Params = dict(
        g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.387
    )

    def ode(self, stimulus=0.0):

        alpha = math.exp(-(self.v + 55.0) / 10.0) - 1.0
        beta = 0.125 * math.exp(-(self.v + 65.0) / 80.0)

        self.d_n = (
            0.1 * (1.0 - self.n) - beta * self.n
            if abs(alpha) <= 1e-7
            else (-0.01 * (self.v + 55.0) / alpha) * (1.0 - self.n) - beta * self.n
        )

        alpha = math.exp(-(self.v + 40.0) / 10.0) - 1.0
        beta = 4.0 * math.exp(-(self.v + 65.0) / 18.0)

        self.d_m = (
            (1.0 - self.m) - beta * self.m
            if abs(alpha) <= 1e-7
            else (-0.1 * (self.v + 40.0) / alpha) * (1.0 - self.m) - beta * self.m
        )

        alpha = 0.07 * math.exp(-(self.v + 65.0) / 20.0)
        beta = 1.0 / (math.exp(-(self.v + 35.0) / 10.0) + 1.0)
        self.d_h = alpha * (1 - self.h) - beta * self.h

        i_na = self.g_Na * pow(self.m, 3) * self.h * (self.v - self.E_Na)
        i_k = self.g_K * pow(self.n, 4) * (self.v - self.E_K)
        i_l = self.g_L * (self.v - self.E_L)

        self.d_v = stimulus - i_na - i_k - i_l
