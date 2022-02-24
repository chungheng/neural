# pylint:disable=invalid-name
from dataclasses import dataclass
import inspect
import ast
import textwrap
from ast import unparse
import typing as tp
from jinja2 import Template
from .. import types as tpe
from .. import errors as err


NUMBA_TEMPLATE = Template(
    """
import numba
import numba.cuda

{% if target == 'cuda' -%}
@numba.cuda.jit
{%- else -%}
@numba.njit
{%- endif %}
def {{ method }}{{signature}}:
    {% if target == 'cuda' -%}
    for {{ idx }} in range(numba.cuda.grid(1), self.shape[0], numba.cuda.gridsize(1)):
    {%- else -%}
    for {{ idx }} in range({{ N }}):
    {%- endif %}
{{ ode_expressions }}
"""
)


class CollectVariables(ast.NodeVisitor):
    def __init__(self):
        self.local_vars = list()

    def visit_Name(self, node: ast.Name) -> tp.Any:
        self.local_vars.append(node.id)


class ReplaceAttr(ast.NodeTransformer):
    def __init__(self, idx_name: str = "_i"):
        self.idx_name = idx_name

    def visit_Name(self, node: ast.Name) -> tp.Any:
        if node.id == "self":
            node = ast.parse(f"self[{self.idx_name}]").body[0].value
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
    if not inspect.isclass(model):
        model = model.__class__

    if not callable(func := getattr(model, method, None)):
        raise err.NeuralCodeGenError(f"Method '{method}' not valid for model {model}")

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
            idx_name=idx_name,
        ).visit(tree)
    )

    # render function source
    func_src = NUMBA_TEMPLATE.render(
        method=method,
        signature=str(inspect.signature(func)),
        ode_expressions=textwrap.indent(unparse(tree.body[0].body), prefix=" " * 8),
        idx=idx_name,
        target=target
    )
    return func_src
