"""Parse BaseModel
"""
import re
import typing as tp
from dataclasses import dataclass
from numbers import Number
import contextlib
import ast
import sys
import inspect
import textwrap
import math
import random
import sympy as sp
import sympy.codegen.cfunctions
import sympy.codegen.ast
from sympy.codegen.ast import Assignment
import sympy.stats
from sympy.printing.latex import LatexPrinter
from .. import types as tpe
from .. import errors as err
from ._ast_transformers import NumPy2SymPy


if sys.version_info < (3, 9):
    from astunparse import unparse
else:
    from ast import unparse


class AlignedEquationPrinter(LatexPrinter):
    """Create alinged LaTeX equations based on parsed model"""

    def _print_list(self, expr) -> str:
        items = []
        for _exp in expr:
            line = self._print(_exp)
            if "&=" not in line:
                line = "&" + line
                items.append(line)
        return r"\begin{aligned} %s \end{aligned}" % r" \\ ".join(items)

    def _print_Equality(self, d) -> str:
        return r"{} &= {}".format(self._print(d.lhs), self._print(d.rhs))

    def _print_Symbol(self, expr, style="plain"):
        if expr.name.startswith("local_"):  # internal variables
            res = super()._print_Symbol(sp.Symbol(expr.name[6:]), style=style)
        else:
            res = super()._print_Symbol(expr, style=style)
        res = re.sub(r"((?:infty|inf|infinity))", r"\\infty", res)

        return res


def print_system_of_equations(list_of_expressions, local_dict=None) -> str:
    """Print system of equations compactly"""
    exprs = list(list_of_expressions)
    for n, l in enumerate(exprs):
        if isinstance(l, str):
            exprs[n] = sp.parsing.parse_expr(l, local_dict=local_dict)
        else:
            continue
    return AlignedEquationPrinter().doprint(exprs)


class RerouteGetattrSetattr:
    """A context to reroute BaseModel.__getattr__ to return sympy symbols"""

    def __init__(self, parsedmodel):
        self.model = parsedmodel
        self.old_getattr = None
        self.old_setattr = None

    def __enter__(self):
        self.old_getattr = self.model.model.__class__.__getattr__
        self.old_setattr = self.model.model.__class__.__setattr__

        def temp_getattr(_self, key):
            if key[:2] == "d_":
                return self.model.gstates[key[2:]].sym

            if key in _self.states:
                return self.model.states[key].sym

            if key in _self.params:
                return self.model.params[key].sym

            raise KeyError(f"Attribute {key} not found model gstates/states/params.")

        def temp_setattr(_self, key, val):
            self.model._local_dict[key] = val

        self.model.model.__class__.__getattr__ = temp_getattr
        self.model.model.__class__.__setattr__ = temp_setattr
        return self.model.model

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.model.__class__.__getattr__ = self.old_getattr
        self.model.model.__class__.__setattr__ = self.old_setattr


@dataclass(repr=False)
class Symbol:
    name: str  # original string name in the model
    sym: sp.Symbol  # sympy symbol
    value: sp.Symbol = None  # numerical value
    default: Number = None  # default numerical value

    def __repr__(self):
        return self.sym.__repr__()


class ParsedModel:
    def __init__(self, model: tpe.Model):
        """Parsed Model Specification

        This module takes instances or class definitions of
        :py:class:`neural.Model` and returns a representation of the
        parsed model.

        Parameters:
            model: instance or subclass or `BaseModel`.

        Attributes:
            model: reference to neural.BaseModel instance
            states: a dictionary mapping Model.Default_States variable names to sympy symbols
            gstates: a dictionary mapping Model.Derivates variable names to sympy symbols
            params: a dictionary mapping Model.Default_Params variable names to sympy symbols
            locals: a dictionary mapping local variables in Model.ode to sympy symbols
            raw_exprs: a list of tuple of str specifying (lhs, rhs) of each equality in Model.ode.
              If expression is not an equality in Model.ode, the entry is (None, expr)
            ode: a list of sympy expressions corresponding to every line in Model.ode.
              - This list of expressions is used for code generation and display
            gradients: a dictionary mapping state variables d_state to gradients after evaluating
              the entire Model.ode symbolically.
              -  This dictionary of expressions can be used for computing jacobian

        Methods:
            F(**sub_vars): return the right hand side of dx/dt = F(x) as a column vector. sub_vars
              correspond to substitutions of variables (states, g_states, params, inputs) into
              the expression
            jacobian(**sub_vars): Jacobian of the ode in each variable. sub_vars are used by
              F(**sub_vars) to first create the gradient vector.
            pprint(gradients=False): pretty print the expressions. If gradients=True, the self.gradients
              are printed. Otherwise the self.disp_exprs is printed in the condensed form.
              - Only supported in IPython kernels (IPython, Jupyter Notebook, JupyterLab)
        """
        if inspect.isclass(model):
            model = model()
        self.model = model

        # create sympy variables
        # get the str name of `self` in case it's different
        _self = inspect.getfullargspec(model.ode).args[0]
        func_args = inspect.signature(model.ode)
        self.inputs = {}
        for n, arg in enumerate(func_args.parameters.values()):
            if arg.default == inspect.Parameter.empty:
                raise ValueError(f"Only keyword arguments are supported: '{arg}'")
            self.inputs[arg.name] = Symbol(
                name=arg.name, sym=sp.Symbol(arg.name), default=arg.default
            )
        self.t_sym = sp.Symbol("t")
        self.states = {  # map model state name to sympy function
            key: Symbol(
                name=key,
                sym=sp.Function(key)(self.t_sym),  # pylint:disable=not-callable
                default=default,
                value=model.states[key],
            )
            for key, default in model.Default_States.items()
        }
        self.gstates = {
            key: Symbol(
                name=key,
                sym=sp.Derivative(self.states[key].sym, self.t_sym),
                value=value,
            )
            for key, value in model.gstates.items()
        }
        self.params = {
            key: Symbol(
                name=key, sym=sp.Symbol(key), default=default, value=model.params[key]
            )
            for key, default in model.Default_Params.items()
        }
        self.internals = dict()  # internal variables to be populated during parsing
        self.bounds = {  # map model state name to sympy function
            key: Symbol(name=key, sym=self.states[key].sym, value=value)
            for key, value in model.bounds.items()
        }

        # Parse ODE as AST
        raw_source = inspect.getsource(model.ode)
        tree = ast.parse(textwrap.dedent(raw_source))
        assert isinstance(
            tree.body[0], ast.FunctionDef
        ), f"""Model ODE does not start with function definition:
            {raw_source}
            """
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        # Mutate AST to be SymPy-Compatible
        tree = ast.fix_missing_locations(
            NumPy2SymPy(global_dct=model.ode.__globals__).visit(tree)
        )

        # loop over ast tree nodes
        self.raw_exprs = []
        for n, node in enumerate(tree.body[0].body):
            # unparse expression
            if isinstance(node, ast.Assign):
                tgt = unparse(node.targets)
                src = unparse(node.value)
                if tgt not in self.Variables:
                    self.internals[tgt] = Symbol(
                        name=tgt,
                        sym=sp.Symbol(
                            f"local_{tgt}"
                        ),  # prefix with local to avoid duplicate name
                    )
                self.raw_exprs.append((tgt, src))
            else:
                self.raw_exprs.append((None, src))

        # reroute getattr of model to return sympy symbols defined above
        self.ode = []  # display expressions sympy.Eq(target, source)
        self._local_dict = dict()  # evaluate expressions
        # with self._reroute_getattr_setattr() as model:
        with RerouteGetattrSetattr(self) as model:
            variable_dict = {
                **model.ode.__globals__,
                **{var: val.sym for var, val in self.inputs.items()},
                **{var: val.sym for var, val in self.internals.items()},
                **{_self: model, "sympy": sympy, "Assignment": Assignment},
            }
            for n, (tgt, src) in enumerate(self.raw_exprs):
                _expr = f"sympy.Eq({tgt}, {src})" if tgt is not None else src
                _exec_expr = f"{tgt} = {src}" if tgt is not None else src
                try:
                    self.ode.append(eval(_expr, variable_dict))
                except Exception as e:
                    raise err.NeuralCodeGenError(
                        f"Evaluating Display Expression Failed:\n\t{_expr}"
                    ) from e

                try:
                    exec(_exec_expr, variable_dict, self._local_dict)
                except Exception as e:
                    raise err.NeuralCodeGenError(
                        f"Evaluating Evaluated Expression Failed:\n\t{_exec_expr}"
                    ) from e

        self.gradients = {var: self._local_dict[f"d_{var}"] for var in self.gstates}

    @property
    def Variables(self):
        return {
            **{key: "states" for key in self.states},
            **{f"d_{key}": "gstates" for key in self.gstates},
            **{key: "params" for key in self.params},
            **{key: "inputs" for key in self.inputs},
        }

    def F(self, **sub_vars) -> "sympy.Matrix":
        """Evaluate the RHS of ODE dx/dt = F(x)"""
        variable_dict = {
            **{key: val.sym for key, val in self.states.items()},
            **{f"d_{key}": val.sym for key, val in self.gstates.items()},
            **{key: val.sym for key, val in self.params.items()},
            **{key: val.sym for key, val in self.inputs.items()},
        }
        sub_vars_sp = [
            (variable_dict[str_varname], val) for str_varname, val in sub_vars.items()
        ]
        return sp.Matrix(list(self.gradients.values())).subs(sub_vars_sp)

    def pprint(self, gradients=False):
        """Pretty Print Model in IPython Environments"""
        from IPython.display import Math  # pylint:disable=import-outside-toplevel

        if gradients:
            return Math(
                "=".join(
                    [
                        sp.latex(
                            sp.Matrix([symbol.sym for symbol in self.gstates.values()])
                        ),
                        sp.latex(self.F()),
                    ]
                )
            )
        return Math(print_system_of_equations(self.ode))

    def jacobian(self, **sub_vars):
        """Compute Jacobian of the Model"""
        jacc = self.F(**sub_vars).jacobian(
            [symbol.sym for symbol in self.states.values()]
        )
        return jacc

    def get_lambidified_jacobian(self) -> tp.Callable:
        arguments = [
            val if name != "states" else tuple(self.states.values())
            for name, val in self.inputs.items()
        ]
        jacc_f = sp.lambdify(arguments, self.jacobian())
        return jacc_f

    def __repr__(self):
        return f"Parsed Model of {self.model.__class__.__name__}"

    @contextlib.contextmanager
    def _reroute_getattr_setattr(self):
        """A context to reroute BaseModel.__getattr__ to return sympy symbols"""
        old_getattr = self.model.__class__.__getattr__
        old_setattr = self.model.__class__.__setattr__

        def temp_getattr(_self, key):
            try:
                if key[:2] == "d_":
                    return self.gstates[key[2:]].sym

                if key in _self.states:
                    return self.states[key].sym

                if key in _self.params:
                    return self.params[key].sym

                raise KeyError(
                    f"Attribute {key} not found model gstates/states/params."
                )
            except Exception as e:
                # put back old getattr before raising
                self.model.__class__.__getattr__ = old_getattr
                raise e

        def temp_setattr(_self, key, val):
            try:
                # var = temp_getattr(_self, key)
                self._local_dict[key] = val
            except Exception as e:
                # put back old getattr before raising
                self.model.__class__.__setattr__ = old_setattr
                raise e

        self.model.__class__.__getattr__ = temp_getattr
        self.model.__class__.__setattr__ = temp_setattr
        yield self.model
        self.model.__class__.__getattr__ = old_getattr
        self.model.__class__.__setattr__ = old_setattr
