import pytest
import numpy as np
from neural.basemodel import Model
from helper_funcs import assert_snr
from scipy.integrate import odeint, solve_ivp, ode, trapz
from scipy.interpolate import interp1d
from neural.utils.signal import generate_stimulus
from neural.solver import SOLVERS
from neural.model.neuron import (
    IAF,
    LeakyIAF,
    Rinzel,
    Wilson,
    ConnorStevens,
    HodgkinHuxley,
)



MIN_SNR = 25
DECIMAL = 0


@pytest.fixture()
def data():
    dt = 1e-3
    t = np.arange(0, 1, dt)
    a = 5.0
    y0 = 100.0
    u = generate_stimulus(
        mode="step",
        d_t=dt,
        duration=t.max() + dt,
        support=(0.2, 0.8),
        amplitude=a,
        sigma=5.0,
    )
    global interp_f
    interp_f = interp1d(t, u, kind="zero", fill_value="extrapolate")
    return t, a, y0, u


@pytest.fixture()
def neuron_data():
    dt = 1e-5
    t = np.arange(0, 0.1, dt)
    a = 15.0
    u = generate_stimulus(
        mode="step", d_t=dt, duration=t.max() + dt, support=(0.02, 0.08), amplitude=a
    )
    interp_neuron_f = interp1d(t, u, kind="zero", fill_value="extrapolate")
    return t, a, u, interp_neuron_f


def exp_f(t, y, a):
    return [-a * y]


def exp_inp_f(t, y, a):
    return [-a * y + interp_f(t)]


def exact_solution(t, a, y0):
    return y0 * np.exp(-a * t)


def exact_solution_inp(t, a, y0, u):
    return (
        y0 * np.exp(-a * t)
        + (t[1] - t[0]) * np.convolve(np.exp(-a * t), u, mode="full")[: len(u)]
    )


class Exp(Model):
    Default_Params = dict(a=1.0)
    Default_States = dict(y=0.0)

    def ode(self, I_ext=0.):
        self.d_y = - self.a * self.y

    def jacobian(self, t, states):
        d_y = - self.a * states
        return np.diag(d_y)


class ExpInp(Model):
    Default_Params = dict(a=1.0)
    Default_States = dict(y=0.0)

    def ode(self, I_ext=0.):
        self.d_y = -self.a * self.y + I_ext

    # You can override the Jacobian method by providing your own callable
    # function that returns the jacobian
    def jacobian(self, t, states, I_ext):
        d_y = -self.a * states + I_ext
        return np.diag(d_y)


@pytest.mark.parametrize("solver", ["RK45", "RK23", "DOP853", "LSODA"])
def test_ivp_solver(data, solver):
    t, a, y0, u = data
    exact_sol = exact_solution(t, a, y0)

    ivp_sol = solve_ivp(
        fun=exp_f,
        t_span=(t.min(), t.max()),
        y0=[y0],
        t_eval=t,
        args=(a,),
        method=solver,
        max_step=t[1]-t[0]
    ).y
    # assert_snr(exact_sol, np.squeeze(ivp_sol), MIN_SNR)
    np.testing.assert_almost_equal(exact_sol, np.squeeze(ivp_sol), decimal=DECIMAL)


@pytest.mark.parametrize("solver", SOLVERS)
def test_neural_solver(data, solver):
    t, a, y0, u = data
    exact_sol = exact_solution(t, a, y0)

    # neu = Exp(a=a, y=y0)
    # neural_sol = neu.solve(t, solver=solver, verbose=False)["y"]
    # assert_snr(exact_sol, np.squeeze(neural_sol), MIN_SNR)

    neu = Exp(num=10, a=a, y=y0, solver=solver, solver_kws=dict(max_step=t[1]-t[0]))
    neural_sol = neu.solve(t, verbose=False)["y"]
    assert len(neural_sol) == 10 == neu.num
    # assert_snr(exact_sol, np.squeeze(neural_sol)[0], MIN_SNR)
    np.testing.assert_almost_equal(exact_sol, np.squeeze(neural_sol)[0], decimal=DECIMAL)


@pytest.mark.parametrize("solver", ["vode", "lsoda", "dopri5", "dop853"])
def test_ode_solver(data, solver):
    t, a, y0, u = data
    dt = t[1] - t[0]
    exact_sol = exact_solution(t, a, y0)

    ode_sol = np.zeros_like(t)
    ode_sol[0] = y0
    ode_solver = ode(exp_f).set_integrator(solver)
    ode_solver.set_initial_value([y0], t.min()).set_f_params(a)

    tt = 0
    while ode_solver.successful() and ode_solver.t < t.max():
        ode_sol[tt + 1] = ode_solver.integrate(ode_solver.t + dt)
        tt += 1

    # assert_snr(exact_sol, np.squeeze(ode_sol), MIN_SNR)
    np.testing.assert_almost_equal(exact_sol, ode_sol, decimal=DECIMAL)


@pytest.mark.parametrize("solver", ["RK45", "RK23", "DOP853", "LSODA"])
def test_ivp_solver_inp(data, solver):
    t, a, y0, u = data
    exact_sol = exact_solution_inp(t, a, y0, u)

    ivp_sol = solve_ivp(
        fun=exp_inp_f,
        t_span=(t.min(), t.max()),
        y0=[y0],
        t_eval=t,
        args=(a,),
        method=solver,
        max_step=(t[1] - t[0])
    ).y
    # assert_snr(exact_sol, np.squeeze(ivp_sol), MIN_SNR)
    np.testing.assert_almost_equal(exact_sol, np.squeeze(ivp_sol), decimal=DECIMAL)

@pytest.mark.parametrize("solver", SOLVERS)
def test_neural_solver_inp(data, solver):
    t, a, y0, u = data
    exact_sol = exact_solution_inp(t, a, y0, u)

    neu = ExpInp(a=a, y=y0, solver=solver, solver_kws=dict(max_step=t[1]-t[0]))
    neural_sol = neu.solve(t, I_ext=u, verbose=False)["y"]
    # assert_snr(exact_sol, np.squeeze(neural_sol), MIN_SNR)
    np.testing.assert_almost_equal(exact_sol, np.squeeze(neural_sol), decimal=DECIMAL)

    neu = ExpInp(num=10, a=a, y=y0, solver=solver, solver_kws=dict(max_step=t[1]-t[0]))
    neural_sol = neu.solve(t, I_ext=u, verbose=False)["y"]
    assert len(neural_sol) == 10 == neu.num
    # assert_snr(exact_sol, np.squeeze(neural_sol)[0], MIN_SNR)
    np.testing.assert_almost_equal(exact_sol, np.squeeze(neural_sol)[0], decimal=DECIMAL)

@pytest.mark.parametrize("solver", ["vode", "lsoda", "dopri5", "dop853"])
def test_ode_solver_inp(data, solver):
    t, a, y0, u = data
    dt = t[1] - t[0]
    exact_sol = exact_solution_inp(t, a, y0, u)

    ode_sol = np.zeros_like(t)
    ode_sol[0] = y0
    ode_solver = ode(exp_inp_f).set_integrator(solver, max_step=t[1]-t[0])
    ode_solver.set_initial_value([y0], t.min()).set_f_params(a)

    tt = 0
    while ode_solver.successful() and ode_solver.t < t.max():
        ode_sol[tt + 1] = ode_solver.integrate(ode_solver.t + dt)
        tt += 1

    # assert_snr(exact_sol, np.squeeze(ode_sol), MIN_SNR)
    np.testing.assert_almost_equal(exact_sol, np.squeeze(ode_sol), decimal=DECIMAL)


@pytest.mark.parametrize(
    "Model",
    [IAF, LeakyIAF, Rinzel, Wilson, ConnorStevens, HodgkinHuxley,]
)
def test_model_solver(neuron_data, Model):
    t, a, u, interp_f = neuron_data
    dt = t[1] - t[0]

    # test that the Model runs
    neuron = Model(num=1, solver=SOLVERS.euler)
    euler_res = neuron.solve(t, stimulus=u, verbose=False)