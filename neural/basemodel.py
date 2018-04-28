"""
Base model class for neurons and synapses.
"""
from abc import abstractmethod
from StringIO import StringIO
import numpy as np

try:
    from optimizer import OdeGenerator
except ImportError:
    OdeGenerator = None

class Model(object):
    """
    The base model class.

    This class overrides `__getattr__` and `__setattr__`, and hence allows
    direct access to the subattributes contained in `states` and `params`,
    for example::

    # self.params = {'a': 1., 'b':1.}
    # self.states = {'s':0., 'x':0.}
    self.ds = self.a*(1-self.s) -self.b*self.s

    The child class of `Model` class must define the

    Methods:
        ode: a set of ODEs defining the dynamics of the system.
        update: wrapper function to the numeric solver for each time step.
        post: computation after calling the numerical solver.
        clip: clip the state variables.
        forwardEuler: forward Euler method.

    Class Attributes:
        Default_States (dict):
        Default_Params (dict):
        Default_Inters (dict): Optional.

    Attributes:
        states (dict): the state variables, updated by the ODE.
        params (dict): parameters of the model, can only be set during
            contrusction.
        inters (dict): intermediate variables. Optional, exists only if
            `__class__.defaultInters` is defined.
        gstates (dict): the gradient of the state variables.
        bounds (dict): lower and upper bounds of the state variables.
    """
    def __init__(self, **kwargs):
        """
        Initialize the model.

        Keyword arguments:
            optimize (bool): optimize the `ode` function.
        """
        optimize = kwargs.pop('optimize', False) and (OdeGenerator is not None)

        baseobj = super(Model, self)

        # set time scale, ex. Hodgkin-Huxley uses ms rather than s.
        time_scale = getattr(self.__class__, 'Time_Scale', 1.)
        baseobj.__setattr__('time_scale', time_scale)

        # set state variables and parameters
        baseobj.__setattr__('params', self.__class__.Default_Params.copy())
        baseobj.__setattr__('states', {})
        baseobj.__setattr__('bounds', {})

        states = baseobj.__getattribute__('states')
        bounds = baseobj.__getattribute__('bounds')

        for key, val in self.__class__.Default_States.items():
            if hasattr(val, '__len__'):
                assert len(val) == 3
                states[key] = val[0]
                bounds[key] = tuple(val[1:])
            else:
                states[key] = val
        gstates = {('d_%s' % key):0. for key in self.states}
        baseobj.__setattr__('gstates', gstates)

        # check if intermediate variables are defined.
        baseobj.__setattr__('_gettableAttrs', ['states', 'params', 'gstates'])
        if hasattr(self.__class__, 'Default_Inters'):
            baseobj.__setattr__('inters', self.__class__.Default_Inters.copy())
            self._gettableAttrs.append('inters')

        solver = kwargs.pop('solver', 'forward_euler')
        baseobj.__setattr__('solver', baseobj.__getattribute__(solver))

        # set additional variables
        baseobj.__setattr__('_settableAttrs', self._gettableAttrs[:])
        for key, val in kwargs.items():
            self.__setattr__(key, val)

        # make params unchangable
        self._settableAttrs.remove('params')

        # optimize the ode function
        if optimize:
            if not hasattr(self.__class__, 'ode_opt'):
                self.__class__.optimize()
                # ode_opt = types.MethodType(self.__class__.ode_opt, self, self.__class__)

            self._ode = self.ode
            self.ode = self.ode_opt

    @classmethod
    def optimize(cls):
        if not hasattr(cls, 'ode_opt'):
            sio = StringIO()

            code_gen = OdeGenerator(cls, offset=4, ostream=sio)
            code_gen.generate()
            co = compile(sio.getvalue(), '<string>', 'exec')
            locs  = dict()
            eval(co, globals(), locs)

            ode = locs['ode']

            ode.__doc__ = sio.getvalue()
            # ode = types.MethodType(ode, self, self.__class__)
            del locs
            setattr(cls, 'code_generator', code_gen)
            setattr(cls, 'ode_opt', ode)

    def update(self, d_t, **kwargs):
        """
        Wrapper function for each iteration of update.

        Arguments:
            d_t (float): time steps.

        Notes:
            The signature of the function does not specify _stimulus_
            arguments. However, the developer should provide the stimulus
            to the model, ex. `input` or `spike`. If mulitple stimuli are
            required, the developer could specify them as `input1` and `input2`.
        """
        self.solver(d_t*self.time_scale, **kwargs)
        self.post()
        self.clip()

    @abstractmethod
    def ode(self, **kwargs):
        """
        The set of ODEs defining the dynamics of the model.
        """
        pass

    def post(self):
        """
        Post-computation after each iteration of numerical update.

        For example, the hard reset for the IAF neuron must be implemented here.
        Another usage of this function could be the spike detection for
        conductance-based models.
        """
        pass

    def clip(self):
        """
        Clip the state variables after calling the numerical solver.

        The state varaibles are usually bounded, for example, binding
        varaibles are bounded between 0 and 1. However, numerical sovlers
        might cause the value of state varaibles exceed its bounds. A hard
        clip is forced here to ensure the state variables remain in the
        given bounds.
        """
        for key, val in self.bounds.items():
            self.states[key] = np.clip(self.states[key], val[0], val[1])

    def forward_euler(self, d_t, **kwargs):
        """
        Forward Euler method.

        Arguments:
            d_t (float): time steps.
        """
        self.ode(**kwargs)

        for key in self.states:
            self.states[key] += d_t*self.gstates['d_%s' % key]

    def __setattr__(self, key, value):
        for param in self._settableAttrs:
            if param == key:
                super(Model, self).__setattr__(param, value)
                return
            attr = getattr(self, param)
            if key in attr:
                attr[key] = value
                return
        super(Model, self).__setattr__(key, value)

    def __getattr__(self, key):
        for param in self._gettableAttrs:
            attr = getattr(self, param)
            if key == param:
                return attr
            if key in attr:
                return attr[key]
        return super(Model, self).__getattribute__(key)

    def get_conductance(self):
        """
        Access the conductance of the model.
        """
        return self.gmax*self.s

    conductance = property(get_conductance)
