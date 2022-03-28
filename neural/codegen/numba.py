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
from numba.extending import overload

def slice_or_return(arr, idx):
    pass

@overload(slice_or_return)
def slice_or_return_impl(arr, idx):
    if isinstance(arr, numba.types.Array):
        def imp(arr, idx):
            return arr[idx]
    else:
        def imp(arr, idx):
            return arr
    return imp

{% if target == 'cuda' -%}
@numba.cuda.jit
{%- else -%}
@numba.njit
{%- endif %}
def {{ method }}{{signature}}:
{% if doc_string -%}
{{doc_string | indent(first=True) }}
{%- endif -%}
    {% if target == 'cuda' -%}
    for {{ idx }} in range(numba.cuda.grid(1), self.shape[0], numba.cuda.gridsize(1)):
    {%- else -%}
    for {{ idx }} in range(self.shape[0]):
    {%- endif -%}
    {% for var in input_args -%}
    {% if var != 'self' %}
        {{ var }}{{ idx }} = slice_or_return({{ var }}, {{ idx }})
    {%- endif %}
    {%- endfor %}
{{ ode_expressions | indent(width=8, first=True) }}
"""
)


class CollectVariables(ast.NodeVisitor):
    def __init__(self):
        self.local_vars = list()

    def visit_Name(self, node: ast.Name) -> tp.Any:
        self.local_vars.append(node.id)


class UnvectorizeVars(ast.NodeTransformer):
    """Unroll arrays in kernel"""

    def __init__(self, idx_name: str = "_i", input_vars: tp.Iterable[str] = None):
        self.idx_name = idx_name
        self.input_vars = set(input_vars) if input_vars is not None else set()

    def visit_Name(self, node: ast.Name) -> tp.Any:
        if node.id == "self":
            node = ast.parse(f"self[{self.idx_name}]").body[0].value
        elif node.id in self.input_vars:
            node = ast.parse(f"{node.id}{self.idx_name}").body[0].value
        return node


def get_numba_function_source(
    model: tpe.Model, method: str = "ode", target: tp.Literal["cpu", "cuda"] = "cpu"
) -> str:
    """Return a function definition that can be jitted

    Arguments:
        model: model to be generated
        method: method name of the model to be generated
        target: cpu or cuda for numba jit
    """
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
    input_args = list(inspect.signature(func).parameters.keys())
    tree = ast.fix_missing_locations(
        UnvectorizeVars(idx_name=idx_name, input_vars=input_args).visit(tree)
    )

    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise TypeError(
            f"Function Body is not FunctionDef type, but {type(func_def)} type."
        )
    if (doc_string := ast.get_docstring(func_def, clean=True)) is not None:
        doc_string = '"""\n{}\n"""'.format(doc_string)
    else:
        doc_string = None
    if doc_string is not None:
        if (
            isinstance(func_def.body[0], ast.Expr)
            and hasattr(func_def.body[0], "value")
            and isinstance(func_def.body[0].value, ast.Str)
        ):
            del func_def.body[0]
    # render function source
    func_src = NUMBA_TEMPLATE.render(
        method=method,  # name of the method to generate
        doc_string=doc_string,  # docstring
        signature=str(inspect.signature(func)),  # function call signature
        input_args=[var for var in inspect.signature(func).parameters],
        ode_expressions=unparse(func_def.body),  # ode definition body
        idx=idx_name,  # str variable to index each model component
        target=target,  # cpu or cuda
    )
    return func_src
