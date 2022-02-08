TEMPLATE = """
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

struct Parameters {
    {%- for key in params %}
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
"""

MAIN_TEMPLATE = """
States states;
Derivatives gstates;
Parameters params;

/* import data */
{%- for key in states %}
states.{{ key }} = g_{{ key }}[i];
{%- endfor %}
{%- for key in params %}
{{ float_type }} {{ key.upper() }} = g_{{ key }}[i];
{%- endfor %}
{%- for (key, ftype, isArray) in ode_signature %}
{%- if isArray %}
{{ ftype }} {{ key }} = g_{{ key }}[i];
{%-  endif %}
{%- endfor %}

{% macro call_ode(states=states, gstates=gstates) -%}
ode(states, gstates
    {%- for key in params -%}
    , {{ key.upper() }}
    {%- endfor -%}
    {%- for (key, _, _) in ode_signature -%}
    , {{ key }}
    {%- endfor -%}
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
"""

FORWARD_TEMPLATE = """
"""
