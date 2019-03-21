"""
Base model class for neurons and synapses.
"""
from __future__ import print_function
from abc import abstractmethod
from collections import OrderedDict
from numbers import Number
import sys

from six import StringIO, with_metaclass
import numpy as np

PY2 = sys.version_info[0] == 3
PY3 = sys.version_info[0] == 3

if PY2:
    from inspect import getargspec as _getfullargspec
    varkw = 'keywords'
if PY3:
    from inspect import getfullargspec as _getfullargspec
    varkw = 'varkw'

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
    from .codegen.optimizer import OdeGenerator
except ImportError:
    OdeGenerator = None

try:
    import pycuda
    import pycuda.gpuarray as garray
    from pycuda.tools import dtype_to_ctype
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    from .codegen.cuda import CudaKernelGenerator, get_func_signature
except ImportError:
    CudaKernelGenerator = None
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

class ModelMetaClass1(type):
    def __new__(cls, clsname, bases, dct):

        defaults = dct.get('Defaults', dict())
        bounds = dict()

        # extract bound from defaults
        for key, val in defaults.items():
            if hasattr(val, '__len__'):
                assert len(val) == 3, "Variable {} ".format(key) + \
                    "should be a scalar of a iterable of 3 elements " + \
                    "(initial value, upper bound, lower bound), " + \
                    "but {} is given.".format(val)
                defaults[key] = val[0]
                bounds[key] = val[1:]

        dct['Default_Bounds'] = bounds

        # extract variables from member functions
        func_list = [x for x in ['ode', 'post'] if x in dct]

        variables = {}
        locals = {}
        for key in func_list:
            _vars, _locals = analyze_variable(dct[key], defaults, variables)
            variables.update(_vars)
            locals[key] = _locals

        dct['Locals'] = locals
        dct['Variables'] = variables

        for attr in ['Inters', 'Params', 'States']:
            dct["Default_{}".format(attr)] = dict()
        for key, val in variables.items():
            if val == 'inputs':
                continue
            attr = "Default_{}".format(val.title())
            dct[attr][key] = defaults[key]

        if 'Time_Scale' not in dct:
            dct['Time_Scale'] = 1.

        if clsname == 'Model':
            solvers = dict()
            for key, val in dct.items():
                if callable(val) and hasattr(val, '_solver_names'):
                    for name in val._solver_names:
                        solvers[name] = val.__name__
            dct['solver_alias'] = solvers

        return super(ModelMetaClass1, cls).__new__(cls, clsname, bases, dct)

class ModelMetaClass(type):
    def __new__(cls, clsname, bases, dct):
        bounds = dict()
        states = dict()
        variables = dict()
        if 'Default_States' in dct:
            for key, val in dct['Default_States'].items():
                if hasattr(val, '__len__'):
                    bounds[key] = val[1:]
                    states[key] = val[0]
                else:
                    states[key] = val
                variables[key] = 'states'
        dct['Default_Bounds'] = bounds
        dct['Default_States'] = states

        for attr in ('Default_Params', 'Default_Inters'):
            if attr not in dct:
                dct[attr] = dict()
            attr_lower = attr[8:].lower()
            variables.update({key: attr_lower for key in dct[attr].keys()})

        dct['Variables'] = variables

        if 'Time_Scale' not in dct:
            dct['Time_Scale'] = 1.

        if clsname == 'Model':
            solvers = dict()
            for key, val in dct.items():
                if callable(val) and hasattr(val, '_solver_names'):
                    for name in val._solver_names:
                        solvers[name] = val.__name__
            dct['solver_alias'] = solvers

        inputs = dict()
        func_list = [x for x in ['ode', 'post'] if x in dct]
        for key in func_list:
            argspec = _getfullargspec(dct[key])
            if argspec.defaults is None:
                continue
            if argspec.varargs is not None:
                raise TypeError("Variable positional argument is not allowed" \
                                " in {}.{}".format(clsname, key))
            if getattr(argspec, varkw, None) is not None:
                raise TypeError("Variable keyword argument is not allowed in" \
                                " {}.{}".format(clsname, key))
            for val, key in zip(argspec.defaults[::-1], argspec.args[::-1]):
                inputs[key] = val
        dct['Inputs'] = inputs

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
        Variables (dict):
        Inputs (dict):

    Attributes:
        states (dict): the state variables, updated by the ODE.
        params (dict): parameters of the model, can only be set during
            contrusction.
        inters (dict): intermediate variables.
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
        solver = kwargs.pop('solver', 'forward_euler')
        float = kwargs.pop('float', np.float32)
        callback = kwargs.pop('callback', [])

        # set state variables and parameters
        self.params = self.Default_Params.copy()
        self.states = self.Default_States.copy()
        self.bounds = self.Default_Bounds.copy()
        self.inters = self.Default_Inters.copy()

        # set additional variables
        for key, val in kwargs.items():
            attr = self.Variables.get(key, None)
            if attr is None:
                raise AttributeError("Unrecognized variable '{}'".format(key))
            dct = getattr(self, attr)
            if type(val) in (list, tuple):
                val = np.asarray(val)
            dct[key] = val

        self.initial_states = self.states.copy()
        self.initial_inters = self.inters.copy()
        self.gstates = {key:0. for key in self.states}

        # set numerical solver
        solver = self.solver_alias[solver]
        self.solver = getattr(self, solver)

        # optimize the ode function
        if optimize:
            if not hasattr(self.__class__, 'ode_opt'):
                self.__class__.optimize()

            # ode_opt = types.MethodType(self.__class__.ode_opt, self, self.__class__)
            self._ode = self.ode
            self.ode = self.ode_opt

        self._update = self._cpu_update
        self.callbacks = []
        self.add_callback(callback)

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

    def _cuda_free(self):
        if hasattr(self, 'cuda') and hasattr(self.cuda, 'data'):
            for key in self.cuda.data.keys():
                self.cuda.data[key].gpudata.free()
        if hasattr(self, 'cuda') and hasattr(self.cuda, 'seed'):
            self.cuda.seed.free()

    def _allocate_cuda_memory(self, key):
        """
        allocate GPU memroy for variable
        """
        if key in self.cuda.data and len(self.cuda.data[key]) != self.cuda.num:
            del self.cuda.data[key]

        if key not in self.cuda.data:
            array = garray.empty(self.cuda.num, dtype=self.cuda.dtype)
            self.cuda.data[key] = array

    def cuda_reset(self, **kwargs):
        """
        reset the gpu data.

        Reset the GPU data to default values.

        Arguments:
            kwargs (dict): keyward arguments.
        """
        params = []
        for key, attr in self.Variables.items():
            dct = getattr(self, attr)
            val = kwargs.pop(key, dct[key])

            # allocate GPU memory
            if attr != 'params':
                self._allocate_cuda_memory(key)
            elif hasattr(val, '__len__'): # params with __len__
                self._allocate_cuda_memory(key)
                params.append(key)

            if isinstance(val, np.ndarray):
                if val.dtype != self.cuda.dtype:
                    val = val.astype(self.cuda.dtype)
                drv.memcpy_htod(self.cuda.data[key].gpudata, val)
            elif isinstance(val, garray.GPUArray):
                if attr == 'params':
                    assert val.dtype == self.cuda.dtype
                    self.cuda.data[key] = val
                    continue
                if val.dtype != self.cuda.dtype:
                    val = val.get()
                    val = val.astype(self.cuda.dtype)
                    drv.memcpy_htod(self.cuda.data[key].gpudata, val)
                else:
                    drv.memcpy_dtod(self.cuda.data[key].gpudata, val.gpudata)
            elif isinstance(val, Number):
                if attr == 'params':
                    self.params[key] = val
                    continue
                self.cuda.data[key].fill(val)
            else:
                raise TypeError("Invalid {0} variable: {1}".format(attr, key))
        return params

    def cuda_compile(self, **kwargs):
        """
        compile the cuda kernel.

        Keyword Arguments:
            num (int): The number of units for CUDA kernel excution.
            dtype (type): The default type of floating point for CUDA.
        """
        num = kwargs.pop('num', None)
        dtype = kwargs.pop('dtype', np.float32)
        callback = kwargs.pop('callback', [])

        # decide the number of threads:
        for key in self.Variables.keys():
            val = getattr(self, key)
            val = kwargs.get(key, val)
            if hasattr(val, '__len__'):
                _num = len(val)
                num = num or _num
                assert num == _num, 'Mismatch in data size: %s' % key
        else:
            assert num, 'Please give the number of models to run'

        # free up GPU memory
        self._cuda_free()

        self.cuda = SimpleNamespace(num=num, dtype=dtype, data=dict())

        # reset gpu data
        params = self.cuda_reset(**kwargs)

        # assume the rest of kwargs are input-related
        inputs = kwargs.copy()
        for key in inputs.keys():
            assert key in self.Inputs, "Unexpected input '{}'".format(key)

        # generate cuda kernel, a.k.a self.cuda.kernel
        self.get_cuda_kernel(inputs_gdata=inputs, params_gdata=params)

        if self.cuda.has_random:
            self.cuda.seed = drv.mem_alloc(self.cuda.num * 48)
            self.cuda.init_random_seed.prepared_async_call(
                self.cuda.grid,
                self.cuda.block,
                None,
                self.cuda.num,
                self.cuda.seed)

        self._update = self._cuda_update
        self.callbacks = []
        self.add_callback(callback)

    def _cuda_update(self, d_t, **kwargs):
        """

        """
        st = kwargs.pop('st', None)
        args = []
        for key, dtype in zip(self.cuda.args, self.cuda.arg_ctype[2:]):
            if key == 'seed':
                args.append(self.cuda.seed)
                continue

            val = self.cuda.data.get(key, None)
            if val is None:
                val = kwargs[key]
            if hasattr(val, '__len__'):
                assert dtype == 'P', \
                    "Expect GPU array but get a scalar input: %s" % key
                assert val.dtype == self.cuda.dtype, \
                    "GPU array float type mismatches: %s" % key
            else:
                assert dtype != 'P', \
                    "Expect GPU array but get a scalar input: %s" % key

            args.append(val.gpudata if dtype == 'P' else val)

        self.cuda.kernel.prepared_async_call(
            self.cuda.grid,
            self.cuda.block,
            st,
            self.cuda.num,
            d_t*self.Time_Scale,
            *args)

    def cuda_profile(self, **kwargs):
        num = kwargs.pop('num', 1000)
        niter = kwargs.pop('niter', 1000)
        dtype = kwargs.pop('dtype', np.float64)

        self.cuda_compile(num=num, dtype=dtype)

        args = {key: garray.empty(num, dtype) for key in self.cuda.args}

        start = drv.Event()
        end = drv.Event()
        secs = 0.

        for i in range(niter):
            start.record()
            self._cuda_update(0., **args)
            end.record()
            end.synchronize()
            secs += start.time_till(end)

        for key in args:
            args[key].gpudata.free()

        name = self.__class__.__name__
        print('Average run time of {}: {} ms'.format(name, secs/niter))

    def get_cuda_kernel(self, **kwargs):
        assert CudaKernelGenerator is not None

        code_generator = CudaKernelGenerator(self,
            dtype=self.cuda.dtype, **kwargs)
        code_generator.generate()

        try:
            mod = SourceModule(code_generator.cuda_src,
                options = ["--ptxas-options=-v"],
                no_extern_c = code_generator.has_random)
            func = mod.get_function(self.__class__.__name__)
        except:
            lines = code_generator.cuda_src.split('\n')
            num_digits = 1 + int(np.floor(np.log10(len(lines))))
            for index, line in enumerate(lines):
                print("{: >{}}: {}".format(index, num_digits, line))
            raise

        self.cuda.src = code_generator.cuda_src
        self.cuda.args = code_generator.args
        self.cuda.arg_ctype = code_generator.arg_type

        func.prepare(self.cuda.arg_ctype)
        self.cuda.kernel = func

        self.cuda.has_random = code_generator.has_random
        if self.cuda.has_random:
            init_random_seed = mod.get_function('generate_seed')
            init_random_seed.prepare(code_generator.init_random_seed_arg)
            self.cuda.init_random_seed = init_random_seed

        deviceData = pycuda.tools.DeviceData()
        maxThreads = int(np.float(deviceData.registers // func.num_regs))
        maxThreads = int(2**int(np.log(maxThreads) / np.log(2)))
        threadsPerBlock = int(min(256, maxThreads, deviceData.max_threads))
        self.cuda.block = (threadsPerBlock, 1, 1)
        self.cuda.grid = (
            int(min(6 * drv.Context.get_device().MULTIPROCESSOR_COUNT,
                (self.cuda.num-1) / threadsPerBlock + 1)),
            1)

        return func

    def _cpu_update(self, d_t, **kwargs):
        """
        Wrapper function for running solver on CPU.

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
        self.solver(d_t*self.Time_Scale, **kwargs)
        self.post()

    def add_callback(self, callbacks):
        if not hasattr(callbacks, '__len__'):
            callbacks = [callbacks,]
        for func in callbacks:
            assert callable(func)
            self.callbacks.append(func)

    def reset(self, **kwargs):
        """
        reset state and intermediate variables to their initial condition.
        """
        for key, val in kwargs.items():
             assert key in self.Varaibles
             attr = self.Varaibles[key]
             if attr in ['states', 'inters']:
                 key = 'initial_' + key
             dct = getattr(self, attr)
             dct[key] = val
        for key, val in self.initial_states.items():
            self.states[key] = val
        for key, val in self.initial_inters.items():
            self.inters[key] = val
        for key in self.gstates.keys():
            self.gstates[key] = 0.

    def update(self, d_t, **kwargs):
        """
        Wrapper function for each iteration of update.

        ``update`` is a proxy to one of ``_cpu_update`` or ``_cuda_update``.

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
        self._update(d_t*self.Time_Scale, **kwargs)
        for func in self.callbacks:
            func()

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

        if key in ['states', 'params', 'inters', 'bounds']:
            return super(Model, self).__setattr__(key, value)

        if key in self.Variables:
            attr = getattr(self, self.Variables[key])
            attr[key] = value
            return

        super(Model, self).__setattr__(key, value)

    def __getattr__(self, key):
        if 'cuda' in self.__dict__ and key in self.cuda.data:
            return self.cuda.data[key]
        if key[:2] == "d_":
            return self.gstates[key[2:]]

        if key in ['states', 'params', 'inters', 'bounds']:
            return getattr(self, key)

        if key in self.Variables:
            attr = getattr(self, self.Variables[key])
            return attr[key]

        return super(Model, self).__getattribute__(key)
