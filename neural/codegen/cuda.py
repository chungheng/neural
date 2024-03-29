from collections import namedtuple
from six import StringIO, get_function_globals, get_function_code
from six import with_metaclass
from functools import wraps
import random
import numpy as np
from jinja2 import Template
from pycuda.tools import dtype_to_ctype

from pycodegen.codegen import CodeGenerator
from pycodegen.utils import get_func_signature

cuda_src_template = """
{%- if has_random %}
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
{%- endif %}
{% set float_char = 'f' if float_type == 'float' else '' %}
{% for key, val in params.items() -%}
{%- if key not in params_gdata -%}
#define  {{ key.upper() }}\t\t{{ val }}
{% endif -%}
{% endfor -%}
{%- if bounds -%}
{%- for key, val in bounds.items() %}
#define  {{ key.upper() }}_MIN\t\t{{ val[0] }}
#define  {{ key.upper() }}_MAX\t\t{{ val[1] }}
{%- endfor -%}
{% endif %}

{%- if has_random %}
extern "C"{

__global__ void  generate_seed(
    int num,
    curandState *seed
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int nid = tid; nid < num; nid += total_threads)
        curand_init(clock64(), nid, 0, &seed[nid]);

    return;
}
{%- endif %}

struct States {
    {%- for key in states %}
    {{ float_type }} {{ key }};
    {%- endfor %}
};

struct Derivatives {
    {%- for key in derivatives %}
    {{ float_type }} {{ key }};
    {%- endfor %}
};

{% if bounds %}
__device__ void clip(States &states)
{
    {%- for key, val in bounds.items() %}
    states.{{ key }} = fmax{{ float_char }}(states.{{ key }}, {{ key.upper() }}_MIN);
    states.{{ key }} = fmin{{ float_char }}(states.{{ key }}, {{ key.upper() }}_MAX);
    {%- endfor %}
}
{%- endif %}

__device__ void forward(
    States &states,
    Derivatives &gstates,
    {{ float_type }} dt
)
{
    {%- for key in derivatives %}
    states.{{ key }} += dt * gstates.{{ key }};
    {%- endfor %}
}

__device__ int ode(
    States &states,
    Derivatives &gstates
    {%- for key in params_gdata -%}
    ,\n    {{ float_type }} {{ key.upper() }}
    {%- endfor %}
    {%- for (key, ftype, isArray) in ode_signature -%}
    ,\n    {{ ftype }} &{{ key }}
    {%- endfor %}
    {%- if ode_has_random %},\n    curandState &seed{%- endif %}
)
{
    {%- for line in ode_declaration %}
    {{ float_type }} {{ line }};
    {%- endfor %}

{{ ode_src -}}
}

{% if post_src|length > 0 %}
/* post processing */
__device__ int post(
    States &states
    {%- for key in params_gdata -%}
    ,\n    {{ float_type }} {{ key.upper() }}
    {%- endfor %}
    {%- for (key, ftype, isArray) in post_signature -%}
    ,\n    {{ ftype }} &{{ key }}
    {%- endfor %}
)
{
    {%- for line in post_declaration %}
    {{ float_type }} {{ line }};
    {%- endfor %}

{{ post_src -}}
}
{%- endif %}

__global__ void {{ model_name }} (
    int num_thread,
    {{ float_type }} dt
    {%- for key in states -%}
    ,\n    {{ float_type }} *g_{{ key }}
    {%- endfor %}
    {%- if params_gdata|length %}
    {%- for key in params_gdata -%}
    ,\n    {{ float_type }} *g_{{ key }}
    {%- endfor %}
    {%- endif %}
    {%- for (key, ftype, isArray) in ode_signature -%}
    {%- if isArray -%}
    ,\n    {{ ftype }} *g_{{ key }}
    {%-  else -%}
    ,\n    {{ ftype }} {{ key }}
    {%- endif %}
    {%- endfor %}
    {%- if post_src|length > 0 -%}
    {%- for (key, ftype, isArray) in post_signature -%}
    {%- if isArray -%}
    ,\n    {{ ftype }} *g_{{ key }}
    {%-  else -%}
    ,\n    {{ ftype }} {{ key }}
    {%- endif %}
    {%- endfor %}
    {%- endif %}
    {%- if ode_has_random %},\n    curandState *seed{%- endif %}
)
{
    /* TODO: option for 1-D or 2-D */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int nid = tid; nid < num_thread; nid += total_threads) {

        States states;
        Derivatives gstates;

        /* import data */
        {%- for key in states %}
        states.{{ key }} = g_{{ key }}[nid];
        {%- endfor %}
        {%- for key in params_gdata %}
        {{ float_type }} {{ key.upper() }} = g_{{ key }}[nid];
        {%- endfor %}
        {%- for (key, ftype, isArray) in ode_signature %}
        {%- if isArray %}
        {{ ftype }} {{ key }} = g_{{ key }}[nid];
        {%-  endif %}
        {%- endfor %}
        {%- if post_src|length > 0 -%}
        {%- for (key, ftype, isArray) in post_signature %}
        {%- if isArray %}
        {{ ftype }} {{ key }} = g_{{ key }}[nid];
        {%-  endif %}
        {%- endfor %}
        {%-  endif %}

        {% macro call_ode(states=states, gstates=gstates) -%}
        ode(states, gstates
            {%- for key in params_gdata -%}
            , {{ key.upper() }}
            {%- endfor -%}
            {%- for (key, _, _) in ode_signature -%}
            , {{ key }}
            {%- endfor -%}
            {%- if ode_has_random -%}, seed[nid]{%- endif -%}
        );
        {%- endmacro %}

        {%- if solver == 'forward_euler' %}
        {%  if run_step %}{%  endif %}
        /* compute gradient */
        {{ call_ode() }}

        /* solve ode */
        forward(states, gstates, dt);
        {%- elif solver == 'runge_kutta' %}
        States k1, k2, k3, k4, tmp;
        {{ call_ode(gstates='k1') }}
        tmp = states;
        forward(tmp, k1, dt*0.5);
        {{ call_ode(states='tmp', gstates='k2') }}
        tmp = states;
        forward(tmp, k2, dt*0.5);
        {{ call_ode(states='tmp', gstates='k3') }}
        tmp = states;
        forward(tmp, k3, dt);
        {{ call_ode(states='tmp', gstates='k4') }}

        forward(states, k1, dt/6.);
        forward(states, k2, dt/3.);
        forward(states, k3, dt/3.);
        forward(states, k4, dt/6.);
        {%- endif %}

        {% if bounds -%}
        /* clip */
        clip(states);
        {%- endif %}

        {% if post_src|length > 0 -%}
        /* post processing */
        post(states
            {%- for key in params_gdata -%}
            , {{ key.upper() }}
            {%- endfor -%}
            {%- for (key, _, _) in post_signature -%}
            , {{ key }}
            {%- endfor -%}
        );
        {%- endif %}

        /* export data */
        {%- for key in states %}
        g_{{ key }}[nid] = states.{{ key }};
        {%- endfor %}
    }

    return;
}

{% if has_random %}}{%- endif %}
"""
CUDASrc = namedtuple("CUDASrc", ("src", "has_random", "args", "variables"))


class MetaClass(type):
    def __new__(cls, clsname, bases, dct):
        py2cu = dict()
        for key, val in dct.items():
            if callable(val) and hasattr(val, "_source_funcs"):
                for func in val._source_funcs:
                    py2cu[func] = val
        dct["pyfunc_to_cufunc"] = py2cu
        return super(MetaClass, cls).__new__(cls, clsname, bases, dct)


class CudaGenerator(with_metaclass(MetaClass, CodeGenerator)):
    def __init__(self, model, func, dtype, **kwargs):
        self.dtype = dtype
        self.model = model
        self.func = func

        self.float_char = "f" if self.dtype == "float" else ""

        self.params_gdata = kwargs.pop("params_gdata", [])
        self.inputs = kwargs.pop("inputs", dict())

        self.variables = []
        self.has_random = False
        self.func_globals = get_function_globals(self.func)

        CodeGenerator.__init__(
            self, self.func, newline=";\n", offset=4, ostream=StringIO(), **kwargs
        )

        _, self.signature, self.kwargs = self.extract_signature(self.func)
        self.generate()

        self.args = self.process_signature()
        self.src = self.ostream.getvalue()

    def process_signature(self):
        new_signature = []
        for key in self.signature:
            val = self.inputs.get(key, None)
            if val is None:
                isArray = True
                dtype = self.dtype
            else:
                val["used"] = True
                isArray = hasattr(val["value"], "__len__")
                if isArray:
                    dtype = dtype_to_ctype(val["value"].dtype)
                else:
                    dtype = self.dtype
            new_signature.append((key, dtype, isArray))
        return new_signature

    def extract_signature(self, func):
        old_signature = get_func_signature(func)
        new_signature = []

        kwargs = None
        for key in old_signature:
            if key == "self":
                continue
            elif key[:2] == "**":
                kwargs = key[2:]
                continue
            elif key[1] == "*":
                raise
            else:
                new_signature.append(key.split("=")[0])
        return old_signature, new_signature, kwargs

    def _post_output(self):
        self.newline = ";\n"
        CodeGenerator._post_output(self)

    def handle_load_attr(self, ins):
        key = ins.argval
        if self.var[-1] == "self":
            if key in self.model.Default_States:
                self.var[-1] = "states.{0}".format(key)
            elif key[:2] == "d_" and key[2:] in self.model.Default_States:
                self.var[-1] = "gstates.{0}".format(key[2:])
            elif key in self.model.Default_Params:
                self.var[-1] = key.upper()
        else:
            self.var[-1] = "{0}.{1}".format(self.var[-1], key)

    def process_jump(self, ins):
        if len(self.jump_targets) and self.jump_targets[0] == ins.offset:
            if len(self.var):
                self.output_statement()
            self.jump_targets.pop()
            self.space -= self.indent
            self.leave_indent = False

            self.var.append("}")
            self.newline = "\n"
            self.output_statement()

    def handle_store_fast(self, ins):
        if ins.argval == self.var[-1]:
            del self.var[-1]
            return
        if ins.argval not in self.variables and ins.argval not in self.signature:
            self.variables.append(ins.argval)
        self.var[-1] = "{0} = {1}".format(ins.argval, self.var[-1])

    def handle_binary_power(self, ins):
        self.var[-2] = "powf({0}, {1})".format(self.var[-2], self.var[-1])
        del self.var[-1]

    def handle_binary_and(self, ins):
        self.var[-2] = "({0} && {1})".format(self.var[-2], self.var[-1])
        del self.var[-1]

    def handle_binary_or(self, ins):
        self.var[-2] = "({0} || {1})".format(self.var[-2], self.var[-1])
        del self.var[-1]

    def handle_call_function(self, ins):
        narg = int(ins.arg)

        args = [] if narg == 0 else list(map(str, self.var[-narg:]))
        func_name = self.var[-(narg + 1)]
        pyfunc = eval(func_name, self.func_globals)
        cufunc = self.pyfunc_to_cufunc.get(pyfunc)
        if cufunc is not None:
            self.var[-(narg + 1)] = cufunc(self, args)
        else:
            temp = ", ".join(args)
            self.var[-(narg + 1)] = "{0}({1})".format(func_name, temp)

        if narg:
            del self.var[-narg:]

    def handle_pop_jump_if_true(self, ins):
        self.jump_targets.append(ins.arg)
        self.enter_indent = True
        self.var[-1] = "if (!{0}) {{".format(self.var[-1])
        self.newline = "\n"

    def handle_pop_jump_if_false(self, ins):
        self.jump_targets.append(ins.arg)
        self.enter_indent = True
        self.var[-1] = "if ({0}) {{".format(self.var[-1])
        self.newline = "\n"

    def handle_jump_forward(self, ins):
        self.leave_indent = True
        self.output_statement()

        target, old_target = ins.argval, self.jump_targets.pop()

        if target != old_target:
            self.newline = "\n"
            self.var.append("} else {")
            self.enter_indent = True
            self.jump_targets.append(target)
        else:
            self.var.append("}")
            self.newline = "\n"
            self.output_statement()

    def handle_return_value(self, ins):
        val = self.var[-1]
        if val is None:
            val = "0"
        self.var[-1] = "return {0}".format(val)

    def _generate_cuda_func(self, func, args):
        return "{0}({1})".format(func, ", ".join(args))

    def _random_func(func):
        """
        A decorator for registering random functions
        """

        @wraps(func)
        def wrap(self, args):
            self.has_random = True
            return func(self, args)

        return wrap

    def _py2cuda(*source_funcs):
        def wrapper(func):
            func._source_funcs = source_funcs
            return func

        return wrapper

    @_py2cuda(np.abs)
    def _np_abs(self, args):
        return self._generate_cuda_func("abs", args)

    @_py2cuda(np.log)
    def _np_log(self, args):
        return self._generate_cuda_func("log" + self.float_char, args)

    @_py2cuda(np.exp)
    def _np_exp(self, args):
        return self._generate_cuda_func("exp" + self.float_char, args)

    @_py2cuda(np.power)
    def _np_power(self, args):
        return self._generate_cuda_func("pow" + self.float_char, args)

    @_py2cuda(np.cbrt)
    def _np_cbrt(self, args):
        return self._generate_cuda_func("cbrt" + self.float_char, args)

    @_py2cuda(np.sqrt)
    def _np_sqrt(self, args):
        return self._generate_cuda_func("sqrt" + self.float_char, args)

    @_py2cuda(random.gauss, np.random.normal)
    @_random_func
    def _random_gauss(self, args):
        func = "curand_normal(&seed)"

        return "({0}+{1}*{2})".format(args[0], args[1], func)

    @_py2cuda(random.uniform)
    @_random_func
    def _random_uniform(self, args):
        func = "curand_uniform(&seed)"

        if len(args) == 1:
            func = "({0}*{1})".format(args[0], func)
        elif len(args) == 2:
            func = "({0}+({1}-{0})*{2})".format(args[0], args[1], func)

        return func


class CudaKernelGenerator(object):
    tpl = Template(cuda_src_template)

    def __init__(self, model, **kwargs):
        self.dtype = dtype_to_ctype(kwargs.pop("dtype", np.float32))
        self.model = model
        self.solver = model.solver.__name__
        self.float_char = "f" if self.dtype == "float" else ""

        self.params_gdata = kwargs.pop("params_gdata", [])
        dct = kwargs.pop("inputs_gdata", dict())
        self.inputs = {k: {"value": v, "used": False} for k, v in dct.items()}

        cls = self.model.__class__

        self.has_post = np.all([cls.post != base.post for base in cls.__bases__])

        self.generate()

    def generate(self):

        ode = CudaGenerator(
            self.model,
            self.model.ode,
            self.dtype,
            inputs=self.inputs,
            params_gdata=self.params_gdata,
        )

        if self.has_post:
            post = CudaGenerator(
                self.model,
                self.model.post,
                self.dtype,
                inputs=self.inputs,
                params_gdata=self.params_gdata,
            )
        else:
            post = CUDASrc(src="", has_random=False, args=[], variables=[])

        for key, val in self.inputs.items():
            assert val["used"], "Unexpected input argument: '{}'".format(key)

        self.has_random = ode.has_random or post.has_random

        self.cuda_src = self.tpl.render(
            model_name=self.model.__class__.__name__,
            float_type=self.dtype,
            run_step="run_step",
            solver=self.solver,
            states=self.model.states,
            bounds=self.model.bounds,
            params=self.model.params,
            derivatives=self.model.Derivates,
            params_gdata=self.params_gdata,
            has_random=self.has_random,
            ode_src=ode.src,
            ode_signature=ode.args,
            ode_has_random=ode.has_random,
            ode_declaration=ode.variables,
            post_src=post.src,
            post_signature=post.args,
            post_declaration=post.variables,
        )

        self.args = list(self.model.states.keys()) + self.params_gdata

        self.arg_type = "i" + self.dtype[0] + "P" * len(self.args)

        for key, dtype, flag in ode.args + post.args:
            self.args.append(key)
            self.arg_type += "P" if flag else dtype[0]

        if self.has_random:
            self.arg_type += "P"
            self.args.append("seed")
            self.init_random_seed_arg = "iP"
