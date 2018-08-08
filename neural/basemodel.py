"""
Base model class for neurons and synapses.
"""
from abc import abstractmethod
from collections import OrderedDict
from StringIO import StringIO
import numpy as np

try:
    from optimizer import OdeGenerator
except ImportError:
    OdeGenerator = None

try:
    import pycuda
    import pycuda.gpuarray as garray
    from pycuda.tools import dtype_to_ctype
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    from cuda import CudaGenerator, get_func_signature
except ImportError:
    CudaGenerator = None
    pycuda = None

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
        return super(ModelMetaClass, cls).__new__(cls, clsname, bases, dct)

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
    __metaclass__ = ModelMetaClass
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

        gstates = {('d_%s' % key):0. for key in self.states}
        baseobj.__setattr__('gstates', gstates)

        # check if intermediate variables are defined.
        baseobj.__setattr__('_gettableAttrs', ['states', 'params', 'gstates'])
        if hasattr(self.__class__, 'Default_Inters'):
            baseobj.__setattr__('inters', self.__class__.Default_Inters.copy())
            self._gettableAttrs.append('inters')

        solver = kwargs.pop('solver', 'forward_euler')
        solver = self._solver_from_acronym(solver)
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

    def _solver_from_acronym(self, solver):
        if solver in ["forward_euler", "runge_kutta"]:
            return solver
        elif solver == 'rk4':
            return "runge_kutta"
        elif solver == 'forward':
            return "forward_euler"

    def cuda_prerun(self, **kwargs):
        """
        Keyword Arguments:
            num (int): The number of units for CUDA kernel excution.
            dtype (type): The default type of floating point for CUDA.
        """
        num = kwargs.pop('num', None)
        dtype = kwargs.pop('dtype', np.float32)

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
                del self.gdata[key]
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
            min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                (num-1) / self.cuda_kernel.threadsPerBlock + 1),
            1)
        self.cuda_kernel.num = num

        if self.cuda_kernel.has_random:
            self.gdata['seed'] = pycuda.driver.mem_alloc(num * 48)
            self.cuda_kernel.init_random_seed.prepared_async_call(
                self.cuda_kernel.grid,
                self.cuda_kernel.block,
                None,
                self.cuda_kernel.num,
                self.gdata['seed'])

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
                assert val.dtype == self.cuda_kernel.dtype, "Float type mismatches: %s" % key
            args.append(val.gpudata if dtype == 'P' else val)

        self.cuda_kernel.prepared_async_call(
            self.cuda_kernel.grid,
            self.cuda_kernel.block,
            st,
            self.cuda_kernel.num,
            d_t*self.time_scale,
            *args)

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
            print code_generator.cuda_src
            raise

        func.arg_type = code_generator.arg_type
        func.prepare(func.arg_type)

        deviceData = pycuda.tools.DeviceData()
        maxThreads = int(np.float(deviceData.registers // func.num_regs))
        maxThreads = 2**int(np.log(maxThreads) / np.log(2))
        func.threadsPerBlock = np.min([256, maxThreads, deviceData.max_threads])

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

    def runge_kutta(self, d_t, **kwargs):
        """
        Runge Kutta method.

        Arguments:
            d_t (float): time steps.
        """
        state_copy = self.states.copy()

        self.ode(**kwargs)
        k1 = {key[2:]: val*d_t for key, val in self.gstates.items()}

        for key in self.states:
            self.states[key] = state_copy[key] + 0.5*k1[key]
        self.clip()
        self.ode(**kwargs)
        k2 = {key[2:]: val*d_t for key, val in self.gstates.items()}

        for key in self.states:
            self.states[key] = state_copy[key] + 0.5*k2[key]
        self.clip()
        self.ode(**kwargs)
        k3 = {key[2:]: val*d_t for key, val in self.gstates.items()}

        for key in self.states:
            self.states[key] = state_copy[key] + k3[key]
        self.clip()
        self.ode(**kwargs)
        k4 = {key[2:]: val*d_t for key, val in self.gstates.items()}

        for key in self.states:
            incr = (k1[key] + 2.*k2[key] + 2.*k3[key] + k4[key]) / 6.
            self.states[key] = state_copy[key] + incr

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
        if self.is_cuda and hasattr(self, 'gdata') and key in self.gdata:
            return self.gdata[key]

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
