from StringIO import StringIO

import numpy as np
from jinja2 import Template
from pycuda.tools import dtype_to_ctype

from pycodegen.codegen import CodeGenerator
from pycodegen.utils import get_func_signature

cuda_src_template = """
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
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

extern "C"{

__global__ void  generate_seed(
    int num,
    curandState *seed
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = tid; i < num; i += total_threads)
        curand_init(clock64(), i, 0, &seed[i]);

    return;
}
{%- endif %}

struct States {
    {%- for key in states %}
    {{ float_type }} {{ key }};
    {%- endfor %}
};
{% if inters %}
struct Inters {
    {%- for key in inters %}
    {{ float_type }} {{ key }};
    {%- endfor %}
};
{% endif %}

{%- if bounds %}
__device__ void clip(States &states)
{
    {%- for key, val in bounds.items() %}
    states.{{ key }} = fmaxf(states.{{ key }}, {{ key.upper() }}_MIN);
    states.{{ key }} = fminf(states.{{ key }}, {{ key.upper() }}_MAX);
    {%- endfor %}
}
{%- endif %}

__device__ void forward(
    States &states,
    States &gstates,
    {{ float_type }} dt
)
{
    {%- for key in states %}
    states.{{ key }} += dt * gstates.{{ key }};
    {%- endfor %}
}

__device__ int ode(
    States &states,
    States &gstates
    {%- if inters %},\n    Inters &inters{%- endif %}
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
    {%- if inters %},\n    Inters &inters{%- endif %}
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
    {%- if inters %}
    {%- for key in inters -%}
    ,\n    {{ float_type }} *g_{{ key }}
    {%- endfor %}
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
    if (tid > num_thread)
        return;

    States states, gstates;
    {%- if inters %}
    Inters inters;
    {% endif %}

    /* import data */
    {%- for key in states %}
    states.{{ key }} = g_{{ key }}[tid];
    {%- endfor %}
    {%- if inters %}
    {%- for key in inters %}
    inters.{{ key }} = g_{{ key }}[tid];
    {%- endfor %}
    {%- endif %}
    {%- for key in params_gdata %}
    {{ float_type }} {{ key.upper() }} = g_{{ key }}[tid];
    {%- endfor %}
    {%- for (key, ftype, isArray) in ode_signature %}
    {%- if isArray %}
    {{ ftype }} {{ key }} = g_{{ key }}[tid];
    {%-  endif %}
    {%- endfor %}
    {%- if post_src|length > 0 -%}
    {%- for (key, ftype, isArray) in post_signature %}
    {%- if isArray %}
    {{ ftype }} {{ key }} = g_{{ key }}[tid];
    {%-  endif %}
    {%- endfor %}
    {%-  endif %}

    {%  if run_step %}{%  endif %}
    /* compute gradient */
    ode(states, gstates
        {%- if inters %}, inters{%  endif %}
        {%- for key in params_gdata -%}
        , {{ key.upper() }}
        {%- endfor -%}
        {%- for (key, _, _) in ode_signature -%}
        , {{ key }}
        {%- endfor -%}
        {%- if ode_has_random -%}, seed[tid]{%- endif -%}
    );

    /* solve ode */
    forward(states, gstates, dt);

    {% if bounds -%}
    /* clip */
    clip(states);
    {%- endif %}

    {%- if post_src|length > 0 -%}
    /* post processing */
    post(states
        {%- if inters %}, inters{%  endif %}
        {%- for (key, _, _) in post_signature -%}
        , {{ key }}
        {%- endfor -%}
    );
    {%- endif %}

    /* export data */
    {%- for key in states %}
    g_{{ key }}[tid] = states.{{ key }};
    {%- endfor %}
    {%- if inters %}
    {%- for key in inters %}
    g_{{ key }}[tid] = inters.{{ key }};
    {%- endfor %}
    {% endif %}

    return;
}

{% if has_random %}}{%- endif %}
"""

class CudaGenerator(CodeGenerator):
    def __init__(self, model, **kwargs):
        self.dtype = dtype_to_ctype(kwargs.pop('dtype', np.float32))
        self.model = model
        self.params_gdata = kwargs.pop('params_gdata', [])
        self.inputs_gdata = kwargs.pop('inputs_gdata', dict())
        self.variables = []
        self.has_random = False
        self.ode_has_random = False
        self.post_has_random = False
        self.ode_local_variables = None
        self.post_local_variables = None
        self.ode_args = None
        self.post_args = None

        self.ode_src = StringIO()
        self.post_src = StringIO()
        cls = self.model.__class__
        self.has_post = np.all([cls.post != base.post for base in cls.__bases__])

        CodeGenerator.__init__(self, model.ode.func_code, newline=';\n',
                offset=4, ostream=self.ode_src, **kwargs)

        self.tpl = Template(cuda_src_template)

    def generate(self, instructions=None):
        if self.has_post and not len(self.post_src.getvalue()):
            self.generate_post()
        if not len(self.ode_src.getvalue()):
            self.generate_ode()

        self.cuda_src = self.tpl.render(
            model_name=self.model.__class__.__name__,
            float_type=self.dtype,
            run_step='run_step',
            states=self.model.states,
            bounds=self.model.bounds,
            params=self.model.params,
            inters=getattr(self.model, 'inters', None),
            params_gdata=self.params_gdata,
            has_random=self.has_random,
            ode_src=self.ode_src.getvalue(),
            ode_signature=self.ode_args,
            ode_has_random=self.ode_has_random,
            ode_declaration=self.ode_local_variables,
            post_src=self.post_src.getvalue(),
            post_signature=self.post_args,
            post_declaration=self.post_local_variables)

        self.arg_type = 'i' + self.dtype[0] + \
            'P' * len(self.model.states) + \
            'P' * len(getattr(self.model, 'inters', [])) + \
            'P' * len(self.params_gdata)

        if self.ode_args is not None:
            self.arg_type += ''.join(['P' if flag else dtype[0] for _, dtype, flag in self.ode_args])
        if self.post_args is not None:
            self.arg_type += ''.join(['P' if flag else dtype[0] for _, dtype, flag in self.post_args])
        if self.has_random:
            self.arg_type += 'P'
            self.init_random_seed_arg = 'iP'
        # print "%s" % self.cuda_src
        # print self.arg_type

    def generate_ode(self):
        _, self.signature, self.kwargs = self.extract_signature(self.model.ode)
        self.variables = []
        self.has_random = False
        self.ostream = self.ode_src
        self.instructions = self.disassemble(self.model.ode.func_code)
        super(CudaGenerator, self).generate()
        self.ode_local_variables = self.variables[:]
        self.ode_args = self.process_signature()
        self.ode_has_random = self.has_random
        self.has_random = self.has_random or self.ode_has_random

    def generate_post(self):
        _, self.signature, self.kwargs = self.extract_signature(self.model.post)
        self.variables = []
        self.has_random = False
        self.ostream = self.post_src
        self.instructions = self.disassemble(self.model.post.func_code)
        super(CudaGenerator, self).generate()
        self.post_local_variables = self.variables[:]
        self.post_args = self.process_signature()
        self.post_has_random = self.has_random
        self.has_random = self.has_random or self.post_has_random

    def process_signature(self):
        new_signature = []
        for key in self.signature:
            val = self.inputs_gdata.get(key, None)
            if val is None:
                isArray = True
                dtype = self.dtype
            else:
                isArray = hasattr(val, '__len__')
                dtype = val.dtype if isArray else type(val)
                dtype = dtype_to_ctype(dtype)
            new_signature.append((key, dtype, isArray))
        return new_signature

    def extract_signature(self, func):
        old_signature = get_func_signature(func)
        new_signature = []

        kwargs = None
        for key in old_signature:
            if key == 'self':
                continue
            elif key[:2] == '**':
                kwargs = key[2:]
                continue
            elif key[1] == '*':
                raise
            else:
                new_signature.append(key.split('=')[0])
        return old_signature, new_signature, kwargs

    def _post_output(self):
        self.newline = ';\n'
        CodeGenerator._post_output(self)

    def handle_load_attr(self, ins):
        key = ins.arg_name
        if self.var[-1] == 'self':
            if key in self.model.Default_States:
                self.var[-1] = "states." + key
            elif key[:2] == 'd_' and key[2:] in self.model.Default_States:
                self.var[-1] = "gstates." + key[2:]
            elif key in self.model.Default_Params:
                self.var[-1] = key.upper()
            elif hasattr(self.model, 'Default_Inters') and key in self.model.Default_Inters:
                self.var[-1] = "inters." + key
        else:
            self.var[-1] += ".%s" % key

    def process_jump(self, ins):
        if ins.jump == ">>":
            if len(self.jump_targets) and self.jump_targets[0] == ins.addr:
                if len(self.var):
                    self.output_statement()
                self.jump_targets.pop()
                self.space -= self.indent
                self.leave_indent = False

                self.var.append('}')
                self.newline = '\n'
                self.output_statement()

    def handle_store_fast(self, ins):
        if ins.arg_name == self.var[-1]:
            del self.var[-1]
            return
        if ins.arg_name not in self.variables and ins.arg_name not in self.signature:
            self.variables.append(ins.arg_name)
        self.var[-1] = ins.arg_name + ' = ' + self.var[-1]

    def handle_binary_power(self, ins):
        self.var[-2] = "powf(%s, %s)" % (self.var[-2], self.var[-1])
        del self.var[-1]

    def handle_binary_and(self, ins):
        self.var[-2] = "(%s && %s)" % (self.var[-2], self.var[-1])
        del self.var[-1]

    def handle_binary_or(self, ins):
        self.var[-2] = "(%s || %s)" % (self.var[-2], self.var[-1])
        del self.var[-1]

    def handle_call_function(self, ins):
        narg = int(ins.arg)

        # hacky way to handle keyword arguments
        if self.kwargs and self.var[-(narg+1)] == (self.kwargs + ".pop"):
            arg = self.var[-narg][1:-1]
            self.var[-(narg+1)] = arg
            new_arg = "%s" % arg
            self.signature.append(new_arg)
        else:
            args = [] if narg == 0 else self.var[-narg:]
            func = self.var[-(narg+1)]
            func = self.pyfunc_to_cufunc(func, args)
            # tmp = "(%s)" % (', '.join(['%s']*narg))
            # tmp = tmp % tuple(self.var[-narg:])
            self.var[-(narg+1)] = func

        if narg:
            del self.var[-narg:]

    def handle_pop_jump_if_true(self, ins):
        self.jump_targets.append(ins.arg)
        self.enter_indent = True
        self.var[-1] = 'if (!%s) {' % self.var[-1]
        self.newline = '\n'

    def handle_pop_jump_if_false(self, ins):
        self.jump_targets.append(ins.arg)
        self.enter_indent = True
        self.var[-1] = 'if (%s) {' % self.var[-1]
        self.newline = '\n'

    def handle_jump_forward(self, ins):
        self.leave_indent = True
        self.output_statement()

        target = int(ins.arg_name.split(' ')[-1])
        old_target = self.jump_targets.pop()

        if target != old_target:
            self.newline = '\n'
            self.var.append("} else {")
            self.enter_indent = True
            self.jump_targets.append(target)
        else:
            self.var.append('}')
            self.newline = '\n'
            self.output_statement()

    def handle_return_value(self, ins):
        val = self.var[-1]
        if val == 'None':
            val = '0'
        self.var[-1] = "return %s" % val

    def pyfunc_to_cufunc(self, func, args):
        seg = func.split('.')
        if seg[0] == 'np' or seg[0] == 'numpy':
            if seg[1] == 'exp':
                func = 'expf'
            elif seg[1] == 'power':
                func = 'powf'
            elif seg[1] == 'cbrt':
                func = 'cbrtf'
            elif seg[1] == 'sqrt':
                func = 'sqrtf'
            elif seg[1] == 'abs':
                func = 'abs'
        elif seg[0] == 'random':
            self.has_random = True
            if seg[1] == 'uniform':
                func = 'curand_uniform(&seed)'
            if seg[1] == 'gauss':
                func = 'curand_normal(&seed)'

            if len(args) == 1:
                func = "(%s*%s)" % (args[0], func)
            elif len(args) == 2:
                func = "({0}+({1}-{0})*{2})".format(args[0], args[1], func)

            return func

        func += "(%s)" % (', '.join(args))
        return func
