# pylint:disable=no-member
from collections import OrderedDict
from collections.abc import Iterable
from numbers import Number
import pycuda.gpuarray as garray
import pycuda.driver as drv
from pycuda.tools import dtype_to_ctype
import numpy as np

from neurokernel.LPU.NDComponents.NDComponent import NDComponent
from .basemodel import Model
from .codegen.neurodriver import NeuroDriverKernelGenerator

def NDComponentFactory(clsname=None, model=None, outputs=None, dtype=np.double):
    """Factory for creating NeuroDriver NDComponent Class

    Keyword Arguments:
        clsname (str): Name of the NDComponent Class
        model (subclass or instance of neural.basemodel.Model): Neural model to be converted to NDComponent
        outputs (str or iterable of str): state variables of `model` to be set as NDComponent's `updates`
        dtype (dtype): data type of the model
    
    Returns:
        An instance of `NeuralNDComponent` that is a subclass of NDComponent
        with the appropriate kernel (created by `NeuroDriverKernelGenerator`).
    
    Raises:
        TypeError: raised if `model` is neither instance or subclass of `Model`
        TypeError: raised if `model`'s states have value that is neither a number
            nor an iterable.
    """
    assert model is not None, "neural_model needs to be specified"
    if isinstance(outputs, str):
        outputs = [outputs]
    if isinstance(model, Model):
        outputs = outputs or list(model.states.keys())
        nd_code_generator = NeuroDriverKernelGenerator(model, dtype=dtype, outputs=outputs)
        clsname = clsname or f'ND{model.__class__.__name__}'
    elif issubclass(model, Model):
        outputs = outputs or list(model.states.keys())
        nd_code_generator = NeuroDriverKernelGenerator(model(), dtype=dtype, outputs=outputs)
        clsname = clsname or f'ND{model.__name__}'
    else:
        raise TypeError

    nd_code_generator.generate()
    cuda_src = nd_code_generator.cuda_src
    has_random = nd_code_generator.has_random

    # make sure accesses and params are in the sequence represented in the cuda kernel
    accesses = list(filter(lambda a: a in model.Inputs, nd_code_generator.args))
    params = list(filter(lambda a: a in model.params, nd_code_generator.args))

    internals = []
    for key,val in model.states.items():
        if isinstance(val, Number):
            internals.append([key, dtype(val)])
        elif isinstance(val, Iterable):
            internals.append([key, dtype(val[0])])
        else:
            raise TypeError
    internals = OrderedDict(internals)
    return type(clsname, (NeuralNDComponent,), {
        'accesses': accesses,
        'params': params,
        'params_defaults':{k:dtype(val) for k,val in model.params.items()},
        'updates': outputs,
        'internals': internals,
        'time_scale': dtype(model.Time_Scale),
        '_cuda_src': cuda_src,
        '_has_rand': has_random
    })

class NeuralNDComponent(NDComponent):
    """Neural-Compatible NDComponent

    A sublcass of NDComponent that is designed to be subclassed by `NDComponentFactory`

    Attributes:
        accesses (list): list of input variables
        updates (list): list of output variables
        params (list): list of parameters
        params_default (dict): default values of the parameters
        internals (OrderedDict): internal variables of the model and initial value
        time_scale (float): scaling factor of the `dt`
    """
    accesses = []    # list
    updates = []     # list
    params = []      # list
    params_defaults = dict() # dict
    internals = OrderedDict() # orderedDict
    time_scale = 1.  # scales dt
    _cuda_src = ''   # cuda source code
    _has_rand = False
    
    def maximum_dt_allowed(self):
        return np.inf

    def __init__(
        self,
        params_dict,
        access_buffers,
        dt,
        LPU_id=None,
        debug=False,
        cuda_verbose=False,
    ):
        if cuda_verbose:
            self.compile_options = ["--ptxas-options=-v", "--expt-relaxed-constexpr"]
        else:
            self.compile_options = ["--expt-relaxed-constexpr"]

        self.debug = debug
        self.LPU_id = LPU_id
        self.num_comps = params_dict[self.params[0]].size
        self.dtype = params_dict[self.params[0]].dtype

        self.dt = dt * self.time_scale
        self.params_dict = params_dict
        self.access_buffers = access_buffers

        self.internal_states = {
            c: garray.zeros(self.num_comps, dtype=self.dtype) + self.internals[c]
            for c in self.internals
        }

        self.inputs = {
            k: garray.empty(self.num_comps, dtype=self.access_buffers[k].dtype)
            for k in self.accesses
        }

        # make all dtypes consistent
        dtypes = {"dt": self.dtype}
        dtypes.update({"state_" + k: self.internal_states[k].dtype for k in self.internals})
        dtypes.update({"param_" + k: self.params_dict[k].dtype for k in self.params})
        dtypes.update({"input_" + k.format(k): self.inputs[k].dtype for k in self.accesses})
        dtypes.update({"output_" + k: self.dtype for k in self.updates})
        self.update_func = self.get_update_func(dtypes)

        if self._has_rand:
            import neurokernel.LPU.utils.curand as curand
            self.randState = curand.curand_setup(self.num_comps, np.random.randint(10000))
            dtypes.update({'rand':self.dtype})

    def run_step(self, update_pointers, st=None):
        for k in self.inputs:
            self.sum_in_variable(k, self.inputs[k], st=st)
        args = [self.internal_states[k].gpudata for k in self.internals] \
            + [self.params_dict[k].gpudata for k in self.params] \
            + [self.inputs[k].gpudata for k in self.accesses] \
            + [update_pointers[k] for k in self.updates]
        if self._has_rand:
            args += [self.randState.gpudata]

        self.update_func.prepared_async_call(
            self.update_func.grid,
            self.update_func.block,
            st,
            self.num_comps,
            self.dt,
            *args
        )

    def get_update_func(self, dtypes):
        assert self._cuda_src is not None, "_cuda_src is None, cannot compile"
                
        from pycuda.compiler import SourceModule

        mod = SourceModule(
            self._cuda_src,
            options=self.compile_options,
            no_extern_c=self._has_rand,
        )
        func = mod.get_function('run_step')
        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}

        func.prepare("i" + np.dtype(self.dtype).char + "P" * (len(type_dict) - 1))
        func.block = (256, 1, 1)
        func.grid = (
            min(6 * drv.Context.get_device().MULTIPROCESSOR_COUNT,
                (self.num_comps - 1) // 256 + 1),
            1,
        )
        return func