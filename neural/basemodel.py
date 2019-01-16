"""
Base model class for neurons and synapses.
"""
from __future__ import print_function
from abc import abstractmethod
from collections import OrderedDict

from six import StringIO, with_metaclass
import numpy as np

try:
    from types import SimpleNamespace
except:
    class SimpleNamespace (object):
        def __init__ (self, **kwargs):
            self.__dict__.update(kwargs)
        def __repr__ (self):
            keys = sorted(self.__dict__)
            items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
            return "namespace({})".format(", ".join(items))
        def __eq__ (self, other):
            return self.__dict__ == other.__dict__

try:
    from optimizer import OdeGenerator
except ImportError:
    OdeGenerator = None

try:
    import pycuda
    import pycuda.gpuarray as garray
    from pycuda.tools import dtype_to_ctype
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    from .cuda import CudaGenerator, get_func_signature
except ImportError:
    CudaGenerator = None
    pycuda = None

def _dict_iadd_(dct_a, dct_b):
    for key in dct_a.keys():
        dct_a[key] += dct_b[key]
    return dct_a

def _dict_add_(dct_a, dct_b, out=None):
    if out is None:
        out = dct_a.copy()
    else:
        for key, val in dct_a.items():
            out[key] = val
    _dict_iadd_(out, dct_b)
    return out

def _dict_add_scalar_(dct_a, dct_b, sal, out=None):
    if out is None:
        out = dct_a.copy()
    else:
        for key, val in dct_a.items():
            out[key] = val
    for key in dct_a.keys():
        out[key] += sal*dct_b[key]
    return out

class ModelMetaClass(type):
    def __new__(cls, clsname, bases, dct):
        bounds = dict()
        states = dict()
        if 'Default_States' in dct:
            for key, val in dct['Default_States'].items():
                if hasattr(val, '__len__'):
                    bounds[key] = val[1:]
                    states[key] = val[0]
                else:
                    states[key] = val
        dct['Default_Bounds'] = bounds
        dct['Default_States'] = states

        if clsname == 'Model':
            solvers = dict()
            for key, val in dct.items():
                if callable(val) and hasattr(val, '_solver_names'):
                    for name in val._solver_names:
                        solvers[name] = val.__name__
            dct['solver_alias'] = solvers

        return super(ModelMetaClass, cls).__new__(cls, clsname, bases, dct)

def register_solver(*args):
    # when there is no args
    if len(args) == 1 and callable(args[0]):
        args[0]._solver_names = [args[0].__name__]
        return args[0]
    else:
        def wrapper(func):
            func._solver_names = [func.__name__] + list(args)
            return func
        return wrapper

class Model(with_metaclass(ModelMetaClass, object)):
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
        Default_States (dict): The default value of the state varaibles.
            Each items represents the name (`key`) and the default value
            (`value`) of one state variables. If `value` is a tuple of three
            numbers, the first number is the default value, and the last two
            are the lower and the upper bound of the state variables.
        Default_Params (dict): The default value of the parameters. Each items
            represents the name (`key`) and the default value (`value`) of one
            parameters.
        Default_Inters (dict): Optional. The default value of the intermediate
            variables. Each items represents the name (`key`) and the default value of one intermediate variable.
            (`value`) of one state variables.
        Default_Bounds (dict): The lower and the upper bound of the state
            variables. It is created through the `ModelMetaClass`.

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
            float (type): The data type of float point.
        """
        optimize = kwargs.pop('optimize', False) and (OdeGenerator is not None)
        float = kwargs.pop('float', np.float32)

        baseobj = super(Model, self)

        # set time scale, ex. Hodgkin-Huxley uses ms rather than s.
        time_scale = getattr(self.__class__, 'Time_Scale', 1.)
        baseobj.__setattr__('time_scale', time_scale)

        cuda = kwargs.pop('cuda', False) and (pycuda is not None)
        baseobj.__setattr__('is_cuda', cuda)

        # set state variables and parameters
        baseobj.__setattr__('params', self.__class__.Default_Params.copy())
        baseobj.__setattr__('states', self.__class__.Default_States.copy())
        baseobj.__setattr__('bounds', self.__class__.Default_Bounds.copy())

        gstates = {key:0. for key in self.states}
        baseobj.__setattr__('gstates', gstates)

        # check if intermediate variables are defined.
        baseobj.__setattr__('_gettableAttrs', ['states', 'params'])
        if hasattr(self.__class__, 'Default_Inters'):
            baseobj.__setattr__('inters', self.__class__.Default_Inters.copy())
            self._gettableAttrs.append('inters')

        # set numerical solver
        solver = kwargs.pop('solver', 'forward_euler')
        solver = self.solver_alias[solver]
        solver = getattr(self, solver)
        baseobj.__setattr__('solver', solver)

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

    def cuda_prerun(self, **kwargs):
        """
        Keyword Arguments:
            num (int): The number of units for CUDA kernel excution.
            dtype (type): The default type of floating point for CUDA.
        """
        num = kwargs.pop('num', None)
        dtype = kwargs.pop('dtype', np.float32)
        callbacks = kwargs.pop('callbacks', [])
        if not hasattr(callbacks, '__len__'):
            callbacks = [callbacks]
        if isinstance(callbacks, tuple):
            callbacks = list(callbacks)
        for func in callbacks:
            assert callable(func)

        # decide the number of threads
        if len(kwargs) > 0:
            for key, val in kwargs.items():
                # assert getattr(self, key)
                if hasattr(val, '__len__'):
                    _num = len(val)
                    num = num or _num
                    assert num == _num, 'Mismatch in data size: %s' % key
        else:
            assert num, 'Please give the number of models to run'

        # reset gpu data
        if hasattr(self, 'gdata'):
            for key in self.gdata.keys():
                if isinstance(self.gdata[key], garray.GPUArray):
                    self.gdata[key].gpudata.free()
                else:
                    self.gdata[key].free()
        self.gdata = {}

        # allocate gpu data for state variables
        for key, val in self.states.items():
            val = kwargs.pop(key, val)
            if isinstance(val, np.ndarray):
                if val.dtype != dtype:
                    val = val.astype(dtype)
                self.gdata[key] = garray.to_gpu(val)
            elif isinstance(val, garray.GPUArray):
                if val.dtype != dtype:
                    val = val.get()
                    val = val.astype(dtype)
                    self.gdata[key] = garray.to_gpu(val)
                else:
                    self.gdata[key] = val.copy()
            elif not hasattr(val, '__len__'):
                self.gdata[key] = garray.empty(num, dtype=dtype)
                self.gdata[key].fill(val)
            else:
                raise TypeError('Unrecognized type for state variables %s' % key)

        # allocate gpu data for intermediate variables
        for key, val in getattr(self, 'inters', dict()).items():
            val = kwargs.pop(key, val)
            if isinstance(val, np.ndarray):
                if val.dtype != dtype:
                    val = val.astype(dtype)
                self.gdata[key] = garray.to_gpu(val)
            elif isinstance(val, garray.GPUArray):
                if val.dtype != dtype:
                    val = val.get()
                    val = val.astype(dtype)
                    self.gdata[key] = garray.to_gpu(val)
                else:
                    self.gdata[key] = val.copy()
            elif not hasattr(val, '__len__'):
                self.gdata[key] = garray.empty(num, dtype=dtype)
                self.gdata[key].fill(val)
            else:
                raise TypeError('Unrecognized type for intermediate variables %s' % key)

        params_gdata = []
        for key, val in getattr(self, 'params', dict()).items():
            val = kwargs.pop(key, None)
            if val is not None:
                params_gdata.append(key)
            if isinstance(val, np.ndarray):
                if val.dtype != dtype:
                    val = val.astype(dtype)
                self.gdata[key] = garray.to_gpu(val)
            elif isinstance(val, garray.GPUArray):
                if val.dtype != dtype:
                    val = val.get()
                    val = val.astype(dtype)
                    self.gdata[key] = garray.to_gpu(val)
                else:
                    self.gdata[key] = val.copy()

        inputs_gdata = kwargs.copy()

        self.cuda_kernel = self.get_cuda_kernel(
            dtype=dtype, inputs_gdata=inputs_gdata, params_gdata=params_gdata)

        self.cuda_kernel.block = (self.cuda_kernel.threadsPerBlock, 1, 1)
        self.cuda_kernel.grid = (
            int(min(6 * drv.Context.get_device().MULTIPROCESSOR_COUNT,
                (num-1) / self.cuda_kernel.threadsPerBlock + 1)),
            1)
        self.cuda_kernel.num = num

        if self.cuda_kernel.has_random:
            self.gdata['seed'] = drv.mem_alloc(num * 48)
            self.cuda_kernel.init_random_seed.prepared_async_call(
                self.cuda_kernel.grid,
                self.cuda_kernel.block,
                None,
                self.cuda_kernel.num,
                self.gdata['seed'])

        self.cuda_kernel.callbacks = callbacks
        self.is_cuda = True

    def cuda_update(self, d_t, **kwargs):
        st = kwargs.pop('st', None)
        args = []
        for key, dtype in zip(self.cuda_kernel.args, self.cuda_kernel.arg_type[2:]):
            val = self.gdata.get(key, None)
            if key == 'seed':
                args.append(val)
                continue
            if val is None:
                val = kwargs[key]
            if hasattr(val, '__len__'):
                assert dtype == 'P', \
                    "Expect GPU array but get a scalar input: %s" % key
                assert val.dtype == self.cuda_kernel.dtype, \
                    "GPU array float type mismatches: %s" % key
            else:
                assert dtype != 'P', \
                    "Expect GPU array but get a scalar input: %s" % key

            args.append(val.gpudata if dtype == 'P' else val)

        self.cuda_kernel.prepared_async_call(
            self.cuda_kernel.grid,
            self.cuda_kernel.block,
            st,
            self.cuda_kernel.num,
            d_t*self.time_scale,
            *args)

        for func in self.cuda_kernel.callbacks:
            func()

    def cuda_profile(self, **kwargs):
        num = kwargs.pop('num', 1000)
        niter = kwargs.pop('niter', 1000)
        dtype = kwargs.pop('dtype', np.float64)

        self.cuda_prerun(num=num, dtype=dtype)

        args = {key: garray.empty(num, dtype) for key in self.cuda_kernel.args}

        start = drv.Event()
        end = drv.Event()
        secs = 0.

        for i in range(niter):
            start.record()
            self.cuda_update(0., **args)
            end.record()
            end.synchronize()
            secs += start.time_till(end)

        for key in args:
            args[key].gpudata.free()

        name = self.__class__.__name__
        print('Average run time of {}: {} ms'.format(name, secs/niter))

    def get_cuda_kernel(self, **kwargs):
        if CudaGenerator is None:
            return
        dtype = kwargs.pop('dtype', np.float32)
        params_gdata = kwargs.pop('params_gdata', [])
        inputs_gdata = kwargs.pop('inputs_gdata', None)
        code_generator = CudaGenerator(self, dtype=dtype,
            inputs_gdata=inputs_gdata, params_gdata=params_gdata, **kwargs)
        code_generator.generate()

        try:
            mod = SourceModule(code_generator.cuda_src,
                options = ["--ptxas-options=-v"],
                no_extern_c = code_generator.has_random)
            func = mod.get_function(self.__class__.__name__)
        except:
            print(code_generator.cuda_src)
            raise

        func.arg_type = code_generator.arg_type
        func.prepare(func.arg_type)

        deviceData = pycuda.tools.DeviceData()
        maxThreads = int(np.float(deviceData.registers // func.num_regs))
        maxThreads = int(2**int(np.log(maxThreads) / np.log(2)))
        func.threadsPerBlock = int(min(256, maxThreads, deviceData.max_threads))

        func.args = []
        for key in self.states.keys():
            func.args.append(key)
        if hasattr(self, 'inters'):
            for key in self.inters.keys():
                func.args.append(key)
        func.args.extend(params_gdata)
        for key in code_generator.ode_args:
            func.args.append(key[0])

        if code_generator.post_args:
            for key in code_generator.post_args:
                func.args.append(key[0])

        func.has_random = code_generator.has_random
        if func.has_random:
            init_random_seed = mod.get_function('generate_seed')
            init_random_seed.prepare(code_generator.init_random_seed_arg)
            func.init_random_seed = init_random_seed
            func.args.append('seed')

        func.dtype = dtype
        func.src = code_generator.cuda_src

        return func

    def update(self, d_t, **kwargs):
        """
        Wrapper function for each iteration of update.

        Arguments:
            d_t (float): time steps.
            kwargs (dict): Arguments for input(s) or other purposes. For
            example, one can use an extra boolean flag to indicate the
            period for counting spikes.

        Notes:
            The signature of the function does not specify _stimulus_
            arguments. However, the developer should provide the stimulus
            to the model, ex. `input` or `spike`. If mulitple stimuli are
            required, the developer could specify them as `input1` and `input2`.
        """
        self.solver(d_t*self.time_scale, **kwargs)
        self.post()

    @abstractmethod
    def ode(self, **kwargs):
        """
        The set of ODEs defining the dynamics of the model.

        TODO: enable using different state varaibles than self.states
        """
        pass


    def _ode_wrapper(self, states=None, gstates=None, **kwargs):
        """
        A wrapper for calling `ode` with arbitrary varaible than `self.states`

        Arguments:
            states (dict): state variables.
            gstates (dict): gradient of state variables.
        """
        if states is not None:
            _states = self.states
            self.states = states
        if gstates is not None:
            _gstates = self.gstates
            self.gstates = gstates

        self.ode(**kwargs)

        if states is not None:
            self.states = _states

        if gstates is not None:
            self.gstates = _gstates
        else:
            return self.gstates

    def post(self):
        """
        Post-computation after each iteration of numerical update.

        For example, the hard reset for the IAF neuron must be implemented here.
        Another usage of this function could be the spike detection for
        conductance-based models.
        """
        pass

    def clip(self, states=None):
        """
        Clip the state variables after calling the numerical solver.

        The state varaibles are usually bounded, for example, binding
        varaibles are bounded between 0 and 1. However, numerical sovlers
        might cause the value of state varaibles exceed its bounds. A hard
        clip is forced here to ensure the state variables remain in the
        given bounds.
        """
        if states is None:
            states = self.states

        for key, val in self.bounds.items():
            states[key] = np.clip(states[key], val[0], val[1])

    def _increment(self, d_t, states, out_states=None, **kwargs):
        """
        Compute the increment of state variables.

        This function is used for advanced numerical methods.

        Arguments:
            d_t (float): time steps.
            states (dict): state variables.
        """
        if out_states is None:
            out_states = states.copy()

        gstates = self._ode_wrapper(states, **kwargs)

        for key in out_states:
            out_states[key] = d_t*gstates[key]

        return out_states

    def _forward_euler(self, d_t, states, out_states=None, **kwargs):
        """
        Forward Euler method with arbitrary varaible than `self.states`.

        This function is used for advanced numerical methods.

        Arguments:
            d_t (float): time steps.
            states (dict): state variables.
        """
        if out_states is None:
            out_states = states.copy()

        self._increment(d_t, states, out_states, **kwargs)

        for key in out_states:
            out_states[key] += states[key]

        self.clip(out_states)

        return out_states

    @register_solver('euler', 'forward')
    def forward_euler(self, d_t, **kwargs):
        """
        Forward Euler method.

        Arguments:
            d_t (float): time steps.
        """
        self.ode(**kwargs)

        for key in self.states:
            self.states[key] += d_t*self.gstates[key]
        self.clip()

    @register_solver('mid')
    def midpoint(self, d_t, **kwargs):
        """
        Implicit Midpoint method.

        Arguments:
            d_t (float): time steps.
        """
        _states = self.states.copy()

        self._forward_euler(0.5*d_t, self.states, _states, **kwargs)
        self._forward_euler(d_t, _states, self.states, **kwargs)

    @register_solver
    def heun(self, d_t, **kwargs):
        """
        Heun's method.

        Arguments:
            d_t (float): time steps.
        """
        incr1 = self._increment(d_t, self.states, **kwargs)
        tmp = _dict_add_(self.states, incr1)
        incr2 = self._increment(d_t, tmp, **kwargs)

        for key in self.states.keys():
            self.states[key] += 0.5*incr1[key] + 0.5*incr2[key]
        self.clip()

    @register_solver('rk4')
    def runge_kutta_4(self, d_t, **kwargs):
        """
        Runge Kutta method.

        Arguments:
            d_t (float): time steps.
        """
        tmp = self.states.copy()

        k1 = self._increment(d_t, self.states, **kwargs)

        _dict_add_scalar_(self.states, k1, 0.5, out=tmp)
        self.clip(tmp)
        k2 = self._increment(d_t, tmp, **kwargs)

        _dict_add_scalar_(self.states, k2, 0.5, out=tmp)
        self.clip(tmp)
        k3 = self._increment(d_t, tmp, **kwargs)

        _dict_add_(self.states, k3, out=tmp)
        self.clip(tmp)
        k4 = self._increment(d_t, tmp, **kwargs)

        for key in self.states.keys():
            incr = (k1[key] + 2.*k2[key] + 2.*k3[key] + k4[key]) / 6.
            self.states[key] += incr
        self.clip()

    def __setattr__(self, key, value):
        if key[:2] == "d_":
            assert key[2:] in self.gstates
            self.gstates[key[2:]] = value
            return

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
        if self.is_cuda and hasattr(self, 'gdata') and key in self.gdata:
            return self.gdata[key]
        if key[:2] == "d_":
            return self.gstates[key[2:]]

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
