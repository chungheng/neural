# pylint:disable=invalid-name
from dataclasses import dataclass
from functools import wraps
import sys, inspect, ast, textwrap
import typing as tp
from jinja2 import Template
from .. import types as tpe
from .. import errors as err

if sys.version_info < (3, 9):
    from astunparse import unparse
else:
    from ast import unparse

NUMBA_TEMPLATE = Template(
    """
import numba
import numba.cuda

{% if target == 'cuda' -%}
@numba.cuda.jit
{%- else -%}
@numba.njit
{%- endif %}
def {{ method }}_numba(
    {{ arguments|join(',\n    ') }}
):
    {% if target == 'cuda' -%}
    for {{ idx }} in range(numba.cuda.grid(1), {{ N }}, numba.cuda.gridsize(1)):
    {%- else -%}
    for {{ idx }} in range({{ N }}):
    {%- endif %}
{{ ode_expressions }}

def {{ method }}{{signature}}:
    {% if target == 'cuda' -%}
    {{ method }}_numba[{{ grid_size }}, {{ block_size }}](
        {{ caller_args |join(',\n        ') }}
    )
    {%- else -%}
    {{ method }}_numba(
        {{ caller_args |join(',\n        ') }}
    )
    {%- endif -%}
"""
)


class CollectVariables(ast.NodeVisitor):
    def __init__(self):
        self.local_vars = list()

    def visit_Name(self, node: ast.Name) -> tp.Any:
        self.local_vars.append(node.id)


class ReplaceAttr(ast.NodeTransformer):
    def __init__(self, name_replacements: dict, idx_name: str = "_i"):
        self.replacements = name_replacements
        self.idx_name = idx_name

    def visit_Attribute(self, node: ast.Attribute) -> tp.Any:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if (old_var := f"self.{node.attr}") in self.replacements:
                ctx = node.ctx
                var_name = self.replacements[old_var]
                node = ast.parse(f"{var_name}[{self.idx_name}]").body[0].value
                node.ctx = ctx
        return node

    def visit_Name(self, node: ast.Name) -> tp.Any:
        if node.id in self.replacements:
            node.id = self.replacements[node.id]  # inputs
        return node


@dataclass
class JittedFunction:
    name: str
    src: str
    args: tp.Dict[str, str]


def get_numba_function_source(
    model: tpe.Model, method: str = "ode", target: tp.Literal["cpu", "cuda"] = "cpu"
) -> JittedFunction:
    """Return a function definition that can be jitted"""
    if not callable(func := getattr(model, method, None)):
        raise err.NeuralCodeGenError(f"Method '{method}' not valid for model {model}")

    inputs = [p for p in inspect.signature(func).parameters if p != "self"]
    replacements = dict()
    for arg in inputs:
        replacements[arg] = f"input_{arg}"
    for arg in model.Default_States:
        replacements[f"self.{arg}"] = f"state_{arg}"
    for arg in model.Default_Params:
        replacements[f"self.{arg}"] = f"param_{arg}"
    for arg in model.Derivates:
        replacements[f"self.d_{arg}"] = f"gstate_{arg}"
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))

    # collect local variables in the source
    vis = CollectVariables()
    vis.visit(tree)
    local_vars = set(vis.local_vars)

    # find a valid local variable name that is not taken by
    # the kernel's local variable
    idx_name = "_i"
    if idx_name in local_vars:
        for n in range(1000):
            if (idx_name := f"_i{n}") not in local_vars:
                break
        else:
            raise err.NeuralCodeGenError(
                "Cannot find valid idx name as all tested variables "
                "have been assigned as local variables in the kernel."
            )

    # replace attributes self.x with states_x/params_x/gstates_x ...
    tree = ast.fix_missing_locations(
        ReplaceAttr(
            name_replacements=replacements,
            idx_name=idx_name,
        ).visit(tree)
    )

    # render function source
    args = list(replacements.values())
    func_src = NUMBA_TEMPLATE.render(
        method=method,
        signature=str(inspect.signature(getattr(model.__class__, method))),
        caller_args=inputs
        + ["*self.states.values()", "*self.params.values()", "*self.gstates.values()"],
        arguments=args,
        ode_expressions=textwrap.indent(unparse(tree.body[0].body), prefix=" " * 8),
        idx=idx_name,
        N=model.num,
        target=target,
        grid_size=1 if target == "cuda" else None,  # FIXME: Not 1
        block_size=1 if target == "cuda" else None,  # FIXME: Not 1
    )
    return JittedFunction(name=method, src=func_src, args=replacements)
