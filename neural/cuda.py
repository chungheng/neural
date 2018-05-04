from StringIO import StringIO

import numpy as np
from jinja2 import Template
from pycuda.tools import dtype_to_ctype

from pycodegen.codegen import CodeGenerator
from pycodegen.utils import get_func_signature

cuda_src_template = """
{% for key, val in preprocessing.items() -%}
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
    );

    /* solve ode */
    forward(states, gstates, dt);

    {% if bounds -%}
    /* clip */
    clip(states);
    {%- endif %}

    {%- if post_src|length > 0 -%}
    /* post processing */
    post(states, inters);
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
"""

class CudaGenerator(CodeGenerator):
    def __init__(self, model, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        self.dtype = dtype_to_ctype(dtype)
        self.model = model
        self.params_gdata = kwargs.pop('params_gdata', [])
        self.inputs_gdata = kwargs.pop('inputs_gdata', dict())
        self.variables = []
        self.ode_variables = None
        self.post_variables = None
        self.old_signature, self.new_signature, self.kwargs = self.process_signature()

        self.ode_src = StringIO()
        self.post_src = StringIO()
        self.define_src = StringIO()
        self.declaration_src = StringIO()
        cls = self.model.__class__
        self.has_post = np.all([cls.post != base.post for base in cls.__bases__])

        CodeGenerator.__init__(self, model.ode.func_code, newline=';\n',
                offset=4, ostream=self.ode_src, **kwargs)

        self.tpl = Template(cuda_src_template)

    def generate_cuda(self):
        if self.has_post and not len(self.post_src.getvalue()):
            self.variables = []
            self.ostream = self.post_src
            instructions = self.disassemble(self.model.post.func_code)
            self.generate(instructions=instructions)
            self.post_variables = self.variables[:]
        if not len(self.ode_src.getvalue()):
            self.variables = []
            self.ostream = self.ode_src
            self.generate()
            self.ode_variables = self.variables[:]

        self.generate_preprocessing()
        self.generate_declaration()

        ode_signature = []
        for key in self.new_signature:
            val = self.inputs_gdata.get(key, None)
            if val is None:
                isArray = True
                dtype = self.dtype
            else:
                isArray = hasattr(val, '__len__')
                dtype = val.dtype if isArray else type(val)
                dtype = dtype_to_ctype(dtype)
            ode_signature.append((key, dtype, isArray))

        self.cuda_src = self.tpl.render(
            float_type=self.dtype,
            bounds=self.model.bounds,
            run_step='run_step',
            inters=getattr(self.model, 'inters', None),
            states=self.model.states,
            ode_signature = ode_signature,
            ode_declaration = self.ode_variables,
            post_declaration = self.post_variables,
            ode_src=self.ode_src.getvalue(),
            post_src=self.post_src.getvalue(),
            model_name=self.model.__class__.__name__,
            params_gdata=self.params_gdata,
            preprocessing=self.model.params)

        self.arg_type = 'i' + self.dtype[0] + \
            'P' * len(self.model.states) + \
            'P' * len(getattr(self.model, 'inters', [])) + \
            'P' * len(self.params_gdata) + \
            ''.join(['P' if flag else dtype[0] for _, dtype, flag in ode_signature])

        # print "%s" % self.cuda_src
        # print self.arg_type

    def process_signature(self):
        old_signature = get_func_signature(self.model.ode)
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
                new_signature.append(key)
        return old_signature, new_signature, kwargs

    def generate_declaration(self):
        for key in self.variables:
            self.declaration_src.write( "    float %s;\n" % str(key) )

    def generate_preprocessing(self):
        for key, val in self.model.params.items():
            self.define_src.write( "#define %s\t\t%s\n" % (str(key), str(val)) )

    def generate(self, instructions=None):
        super(CudaGenerator, self).generate(instructions)
        # remove defaults from signature
        self.new_signature = [x.split('=')[0] for x in self.new_signature]

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
        if ins.arg_name not in self.variables and ins.arg_name not in self.new_signature:
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
            self.new_signature.append(new_arg)
        else:

            func = self.var[-(narg+1)]
            func = self.pyfunc_to_cufunc(func)
            tmp = "(%s)" % (', '.join(['%s']*narg))
            tmp = tmp % tuple(self.var[-narg:])

            self.var[-(narg+1)] = func + tmp

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

    def pyfunc_to_cufunc(self, func):
        seg = func.split('.')
        if seg[0] == 'np' or seg[0] == 'numpy':
            if seg[1] == 'exp':
                return 'expf'
            elif seg[1] == 'power':
                return 'powf'
            elif seg[1] == 'cbrt':
                return 'cbrtf'
        return func
