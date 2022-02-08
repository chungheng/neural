# pylint:disable=invalid-name
import sys, inspect, ast, textwrap
import typing as tp
import numba
from jinja2 import Template
from .. import types as tpe
from .. import errors as err

if sys.version_info < (3, 9):
    from astunparse import unparse
else:
    from ast import unparse


FUNCIFIED_ODE_TEMPLATE = Template(
    """
def _vectorized_func(
    {{ arguments|join(',\n    ') }}
):
    {{ ode_expressions }}
    return {{ returns|join(', ') }}
"""
)


def _is_implemented(cls: tpe.Model, method: str) -> bool:
    if not hasattr(cls, method):
        return False
    try:
        getattr(cls, method)(cls)
        return True
    except (NotImplementedError, TypeError):
        return False
    except:
        return True


def funcify(cls: tpe.Model, method: tp.Literal["ode", "post"] = "ode") -> str:
    """Return a pure functional representation of Model.ode

    This is useful for optimization with numba
    """
    if method not in ["ode", "post"]:
        raise err.NeuralCodeGenError("only .ode() and .post() methods are supported")
    func = getattr(cls, method)
    _self, *inputs = inspect.getfullargspec(func).args
    replacements = dict()
    for arg in inputs:
        replacements[arg] = f"input_{arg}"
    for arg in cls.Default_States:
        replacements[f"{_self}.{arg}"] = f"state_{arg}"
    for arg in cls.Default_Params:
        replacements[f"{_self}.{arg}"] = f"param_{arg}"
    for arg in cls.Derivates:
        replacements[f"{_self}.d_{arg}"] = f"gstate_{arg}"
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))

    class ReplaceAttr(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> tp.Any:
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == _self:
                node = ast.Name(id=replacements[f"{_self}.{node.attr}"], ctx=node.ctx)
            return node

    tree = ast.fix_missing_locations(ReplaceAttr().visit(tree))

    args = list(replacements.values())
    gstate_returns = [k for k in args if k.startswith("gstate_")]
    func_src = FUNCIFIED_ODE_TEMPLATE.render(
        arguments=args,
        ode_expressions=textwrap.indent(unparse(tree.body[0].body), prefix=" " * 4),
        returns=gstate_returns if method == "ode" else [],
    )
    return func_src


def get_vectorized_func(cls: tpe.Model, method: tp.Literal["ode", "post"] = "ode"):
    if method not in ["ode", "post"]:
        raise err.NeuralCodeGenError("only .ode() and .post() methods are supported")
    if not _is_implemented(cls, f"_{method}_vectorized"):
        try:
            dct = dict(_vectorized_func=None)  # defined in funcify_ode
            exec(funcify(cls, method), dct)
            func = dct["_vectorized_func"]
        except Exception as e:
            raise err.NeuralCodeGenError(
                f"Failed to create functional definition of {cls.__name__}.ode"
            ) from e
        return numba.vectorize(nopython=True)(func)
