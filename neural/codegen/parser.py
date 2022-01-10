"""Parse BaseModel
"""
import textwrap
import ast
import typing as tp
import inspect
from datetime import datetime
import sys
import sympy as sp
from sympy.printing.latex import LatexPrinter
from .. import types as tpe

if sys.version < (3, 9):
    import astunparse


class AlignedEquationPrinter(LatexPrinter):
    """Create alinged LaTeX equations based on parsed model"""

    def _print_list(self, expr) -> str:
        items = []
        for _exp in expr:
            items.append(self._print(_exp))

        return r"\begin{aligned} %s \end{aligned}" % r" \\ ".join(items)

    def _print_Equality(self, d) -> str:
        return r"{} &= {}".format(self._print(d.lhs), self._print(d.rhs))


def print_system_of_equations(list_of_expressions, local_dict=None) -> str:
    """Print system of equations compactly"""
    exprs = list(list_of_expressions)
    for n, l in enumerate(exprs):
        if isinstance(l, str):
            exprs[n] = sp.parsing.parse_expr(l, local_dict=local_dict)
        else:
            continue
    return AlignedEquationPrinter().doprint(exprs)


class ParsedModel:
    def __init__(self, model: tpe.Model):
        """Parsed Model Specification

        This module takes instances or class definitions of 
        :py:class:`neural.Model` and returns a representation of the 
        parsed model.

        Parameters:
            model: instance or subclass or `BaseModel`.
        """
        if inspect.isclass(model):
            model = model()

        # reate sympy variables
        t_sym = sp.Symbol("t")
        self.state_vars = {  # map model state name to sympy function
            key: sp.Function(key)(t_sym)  # pylint:disable=not-callable
            for key in model.states
        }
        self.gstate_vars = {
            f"d_{key}": sp.Derivative(self.state_vars[key], t_sym)  # d_{state}
            for key in model.states
        }
        self.param_vars = {
            key: sp.Symbol(key) for key in model.params  # parameters are symbols
        }
        func_args = inspect.getfullargspec(model.ode)
        if func_args.varkw is not None:
            raise ValueError(
                f"Variable Keyword Argument is not supported: '{func_args.varkw}'"
            )
        self.input_vars = {
            key: sp.Symbol(key)
            for key in func_args.args + func_args.kwonlyargs
            if key is not "self"
        }

        # Cleanup ode source
        # 1. replace double with single quote for parsing
        raw_source = inspect.getsource(model.ode).replace('"', "'")

        # 2. parse ast tree from source
        mod = ast.parse(textwrap.dedent(raw_source))
        assert isinstance(mod.body[0], ast.FunctionDef), \
            f"""Model ODE does not start with function definition:
            {raw_source}
            """

        # parse ast tree to generate cleanedup source
        parsed_eqns = []  # list of sympy equalities in string
        parsed_eqns_eval = []  # lsit of sympy equalities to be evaluated

        # store local variables that are not state or params or inputs or
        # gradient states (gstates)
        self.local_vars = dict()

        # loop over ast tree nodes
        for node in mod.body[0].body:
            disp_eqn, eval_eqn = self._process_ast_node(node)
            if disp_eqn is None:
                continue
            parsed_eqns.append(disp_eqn)
            parsed_eqns_eval.append(eval_eqn)

        # Replace all numpy function calls with sympy function calls
        # TODO: This can be optimized to take into account corner cases where
        #  some numpy functions are not understood by sympy
        self.parsed_eqns = [
            e.replace("np.", "sp.").replace("numpy.", "sp.") for e in parsed_eqns
        ]
        self.parsed_eqns_eval = [
            e.replace("np.", "sp.").replace("numpy.", "sp.") for e in parsed_eqns_eval
        ]

        # A dictionary of all local variables to be used when executing
        # the parsed equations
        local_dict = {
            **sp.__dict__,
            **self.state_vars,
            **self.gstate_vars,
            **self.param_vars,
            **self.input_vars,
            **{"t": t_sym, "sp": sp},
            **self.local_vars,
        }

        # Parse str expressions into sympy expressions using variables in
        # local_dict
        self.parsed_exprs = [
            sp.parsing.parse_expr(expr, local_dict=local_dict)
            for expr in self.parsed_eqns
        ]

        # assign parsed variables
        self.model = model

        # Evaluated Expresssions
        self.evaled_exprs = dict()
        for expr in self.parsed_eqns_eval:
            try:
                exec(expr, local_dict, self.evaled_exprs)
            except Exception as e:
                raise Exception(
                    f"Evaluating Expression Failed on Line:\n\t{expr}"
                ) from e

        # Right hand side of the ODE
        self.ode_rhs = sp.Matrix(
            sum(
                [],
                [
                    [
                        val.rhs
                        for name, val in self.evaled_exprs.items()
                        if isinstance(val, sp.Eq) and val.lhs == l
                    ]
                    for _, l in self.gstate_vars.items()
                ],
            )
        )

    def _process_ast_node(self, node: "ast.Node"):
        """Process AST Node and return parsed expression"""
        if not isinstance(node, ast.Assign):
            # only handle ast.Assign for now
            return None, None

        tgt = node.targets
        val = node.value
        if len(tgt) != 1 or not isinstance(tgt[0], ast.Name):
            # only handle assingment to single variable. This is a hacky
            # method to avoid tuple interpretation for `states` input
            # as defined in BaseModel instances.
            return None, None

        # Convert tgt to string
        tgt_str = astunparse.unparse(tgt).strip("\n")
        val_str = astunparse.unparse(val).strip("\n")
        if isinstance(val, ast.Compare):
            val_str = f"sp.Piecewise((1, {val_str}), (0, True))"

        # target variable that is not states/gstats/params should be local
        if tgt_str not in {**self.state_vars, **self.gstate_vars, **self.param_vars}:
            self.local_vars[tgt_str] = sp.Symbol(tgt_str)

        # Every displayed equation is in the form of Equality condition in Sympy
        disp_eqn = "Eq({tgt}, {val})".format(tgt=tgt_str, val=val_str)

        # Parse equations to be evaluated
        # If the assignment is a gradient definition, then we assign
        #   the parsed equation into a variable
        if tgt_str in self.gstate_vars:
            eval_eqn = "ode_update_eqn_{timestamp} = Eq({tgt}, {val})".format(
                timestamp=int(1e6 * datetime.now().timestamp()),
                tgt=tgt_str,
                val=val_str,
            )
        else:
            # otherwise we just assign the expression into the target
            eval_eqn = "{tgt} = {val}".format(tgt=tgt_str, val=val_str)
        return disp_eqn, eval_eqn

    def pprint(self) -> str:
        """Pretty Print Model in IPython Environments"""
        from IPython.display import Math  # pylint:disable=import-outside-toplevel

        return Math(print_system_of_equations(self.parsed_exprs))

    def jacobian(self, sub_params=True):
        """Compute Jacobian of the Model"""
        jacc = self.ode_rhs.jacobian(list(self.state_vars.values()))
        if sub_params:
            return jacc.subs(self.sub_params())
        return jacc

    def sub_params(self) -> tp.Dict:
        """Create array of substitute parameter variables"""
        return [
            (sympy_var, self.model.params[var_name])
            for var_name, sympy_var in self.param_vars.items()
        ]

    def __repr__(self):
        return f"Parsed Model of {self.model.__class__.__name__}"