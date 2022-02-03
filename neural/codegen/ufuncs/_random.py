import random
import sympy as sp
import sympy.stats 

MAPPING = {
    random.random : dict(
        func='sympy.stats.Uniform',
        random_params=dict(), 
        sympy_params=dict(left=0, right=1)
    ),
    random.uniform : dict(
        func='sympy.stats.Uniform',
        random_params=dict(a=0, b=1),
        sympy_params=dict(left=None, right=None),
    ),
    random.triangular : dict(
        func='sympy.stats.Triangular',
        random_params=dict(low=0, high=1, mode=0.5),
        sympy_params=dict(a=0, b=1, c=0.5),
    ),
    random.betavariate : dict(
        func='sympy.stats.Beta',
        random_params=dict(alpha=None, beta=None),
        sympy_params=dict(n=None, alpha=None, beta=None),
    ),
    random.expovariate : dict(
        func='sympy.stats.Exponential',
        random_params=dict(lambd=None),
        sympy_params=dict(rate=None),
    ),
    random.gammavariate : dict(
        func='sympy.stats.Gamma',
        random_params=dict(alpha=None, beta=None),
        sympy_params=dict(k=None, theta=None),
    ),  # (alpha:k, beta:theta)
    random.gauss : dict(
        func='sympy.stats.Normal',
        random_params=dict(mu=None, sigma=None),
        sympy_params=dict(mean=None, std=None),
    ),
    random.lognormvariate : dict(
        func='sympy.stats.LogNormal',
        random_params=dict(mu=None, sigma=None),
        sympy_params=dict(mean=None, std=None),
    ),
    random.normalvariate : dict(
        func='sympy.stats.Normal',
        random_params=dict(mu=None, sigma=None),
        sympy_params=dict(mean=None, std=None),
    ),
    random.vonmisesvariate : dict(
        func='sympy.stats.VonMises',
        random_params=dict(mu=None, kappa=None),
        sympy_params=dict(mu=None, k=None),
    ),
    random.weibullvariate : dict(
        func='sympy.stats.Weibull',
        random_params=dict(alpha=None, beta=None),
        sympy_params=dict(alpha=None, beta=None),
    ),
    random.paretovariate : None,  # pareto call signature is different from that of sympy
    random.randint : None,
    random.choice : None,
    random.randrange : None,
    random.sample : None,
    random.shuffle : None,
    random.choices : None,
}

SUPPORTED = [key.__name__ for key,val in MAPPING.items() if val is not None]