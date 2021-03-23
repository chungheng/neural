import copy
from functools import wraps
import random

import numpy as np
import os
from six import StringIO, get_function_code, get_function_globals
from six import with_metaclass
from sympy import *
import types

from pycodegen.codegen import CodeGenerator
from pycodegen.utils import get_func_signature


class _Variable(object):
    default = {
        "type": None,
        "integral": None,
        "derivative": None,
        "dependencies": set(),
    }

    def __init__(self, **kwargs):
        for key, val in self.default.items():
            if key in kwargs:
                self.__dict__[key] = kwargs.pop(key)
            else:
                self.__dict__[key] = copy.copy(val)
        if len(kwargs):
            raise AttributeError("Invalid attribute: %s" % kwargs.keys()[0])

    def __setattribute__(self, key, val):
        if key not in self.__dict__:
            raise KeyError("Unrecognized key: %s" % key)
        self.__dict__[key] = val


class MetaClass(type):
    def __new__(cls, clsname, bases, dct):
        py2sympy = dict()
        for key, val in dct.items():
            if callable(val) and hasattr(val, "_source_funcs"):
                for func in val._source_funcs:
                    py2sympy[func] = val
        dct["pyfunc_to_sympyfunc"] = py2sympy
        return super(MetaClass, cls).__new__(cls, clsname, bases, dct)


class VariableAnalyzer(CodeGenerator):
    """
    Analyze the variables in a set of ODEs
    """

    def __init__(self, model, **kwargs):
        self.model = model
        self._dependencies = set()

        _, self.signature, self.kwargs = self._extract_signature(self.model.ode)
        with open(os.devnull, "w") as f:
            code = get_function_code(model.ode)
            CodeGenerator.__init__(self, code, ostream=f)
            self.variables = {}
            self.generate()
            self.generate()
        for key, val in self.variables.items():
            if val.integral is None:
                continue
            self.variables[val.integral].dependencies.update(val.dependencies)

    def _extract_signature(self, func):
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

    def handle_call_function(self, ins):
        narg = int(ins.arg)
        args = [] if narg == 0 else [str(x) for x in self.var[-narg:]]
        func_name = self.var[-(narg + 1)]
        self.var[-(narg + 1)] = "{}({})".format(func_name, ",".join(args))

        if narg:
            del self.var[-narg:]

    def handle_store_fast(self, ins):
        """
        symbol1 = rvalue
        """
        key = ins.argval
        if key not in self.variables:
            self.variables[key] = _Variable(type="local")
        self.var[-1] = "{} = {}".format(key, self.var[-1])
        self.variables[key].dependencies.update(self._dependencies)
        self._dependencies = set()

    def handle_load_attr(self, ins):
        """
        ... symbol1.symbol2 ...
        """
        key = ins.argval
        if self.var[-1] == "self":
            if key[:2] == "d_":
                key = key.split("d_")[-1]
                self._set_variable(key, type="state")
            elif key not in self.variables:
                self._set_variable(key, type="parameter")
            if self.variables[key].type != "parameter":
                self._dependencies.add(key)

            self.var[-1] = key
        else:
            self.var[-1] += "." + ins.argval

    def handle_load_fast(self, ins):
        if ins.argval in self.signature or ins.argval in self.variables:
            self._dependencies.add(ins.argval)
        self.var.append(ins.argval)

    def handle_store_attr(self, ins):
        """
        symbol1.symbol2 = rvalue
        """
        key = ins.argval
        if self.var[-1] == "self":
            if key[:2] == "d_":
                key = key.split("d_")[-1]
                self._set_variable(key, type="state")
                if self.var[-2] in self.variables:
                    self.variables[key].derivative = self.var[-2]
                    self.variables[self.var[-2]].integral = key
            elif key not in self.variables or self.variables[key] != "state":
                self._set_variable(key, type="intermediate")
            self.var[-1] = key
            self.variables[key].dependencies.update(self._dependencies)
            self._dependencies = set()

        else:
            self.var[-1] = "{}.{}".format(self.var[-1], ins.argval)
        self.var[-2] = "{} = {}".format(self.var[-1], self.var[-2])
        del self.var[-1]

    def _set_variable(self, variable, **kwargs):
        if variable not in self.variables:
            self.variables[variable] = _Variable()

        for key, val in kwargs.items():
            setattr(self.variables[variable], key, val)

    def to_graph(self, local=False):
        """
        Parameters:
        local (boolean): Include local variables or not.
        """

        # sort out dependencies for local variables
        locals = [k for k, v in self.variables.items() if v.type == "local"]
        flags = dict.fromkeys(locals, False)
        locals = {k: copy.copy(self.variables[k].dependencies) for k in locals}
        for key, val in locals.items():
            if key in val:
                val.remove(key)

        check_is_local = (
            lambda x: x in self.variables and self.variables[x].type != "local"
        )
        for i in range(len(self.variables)):
            flag = True
            for key, val in locals.items():
                flags[key] = all(map(check_is_local, val))
                flag = flag and flags[key]
            if flag:
                break
            for key, val in locals.items():
                _set = set()
                for v in val:
                    if flags.get(v, False):
                        _set.update(locals[v])
                    else:
                        _set.add(v)
                locals[key] = _set

        import pydot

        graph = pydot.Dot(graph_type="digraph", rankdir="LR")

        node_attrs = dict(shape="rect", style="rounded", fontname="sans-serif")
        nodes = {}
        for key in self.signature:
            node = pydot.Node(key, **node_attrs, color="#ff5000")
            nodes[key] = node
            graph.add_node(node)

        for key, val in self.variables.items():
            if val.type == "parameter" or val.integral is not None:
                continue
            if local is False and val.type == "local":
                continue
            node = pydot.Node(key, **node_attrs, color="#ff5000")
            nodes[key] = node
            graph.add_node(node)

        for target, val in self.variables.items():
            if target not in nodes:
                continue
            _set = copy.copy(val.dependencies)
            for v in val.dependencies:
                if local is False and v in locals:
                    _set.remove(v)
                    _set.update(locals[v])
            for source in _set:
                if source in nodes:
                    graph.add_edge(pydot.Edge(source, target, color="#ffbf00"))

        png_str = graph.create_png(prog="dot")

        return png_str


class SympyGenerator(with_metaclass(MetaClass, VariableAnalyzer)):
    def __init__(self, model, **kwargs):

        self.has_random = False
        VariableAnalyzer.__init__(self, model, **kwargs)
        # for attr in ['state', 'parameter', 'input']:
        #     lst = [k for k, v in self.variables.items() if v.type == attr]
        #     lst.sort()
        #     setattr(self, attr+'s', lst)
        self.ode_src = StringIO()
        self.symbol_src = StringIO()
        self.ostream = self.ode_src
        self.equations = []

        for key in dir(self):
            if key[:8] == "_handle_":
                setattr(self, key[1:], getattr(self, key))
        #
        self.generate()
        self.get_symbols()
        self.sympy_dct = {}
        self.latex_src = None
        self.compile_sympy()

        self.to_latex()

    @property
    def sympy_src(self):
        return self.symbol_src.getvalue() + self.ode_src.getvalue()

    def get_symbols(self):
        if self.has_random:
            self.symbol_src.write("N = Function('N')%c" % self.newline)
        self.symbol_src.write("t = Symbol('t')%c" % self.newline)
        for key, val in self.variables.items():
            src = "{0} = Symbol('{0}'){1}".format(key, self.newline)
            self.symbol_src.write(src)
        for key in self.signature:
            src = "{0} = Symbol('{0}'){1}".format(key, self.newline)
            self.symbol_src.write(src)

    def compile_sympy(self):
        try:
            exec(self.sympy_src, globals(), self.sympy_dct)
        except:
            print(self.sympy_src)

    def to_latex(self):
        cond = lambda v: v.type == "state" and v.integral is None
        states = [k for k, v in self.variables.items() if cond(v)]
        states.sort()
        states_src = ",~".join([latex(self.sympy_dct[x]) for x in states])
        params = [k for k, v in self.variables.items() if v.type == "parameter"]
        params.sort()
        params_src = ",~".join([latex(self.sympy_dct[x]) for x in params])
        template_src = r"\mbox{State Variables: }%s\\\mbox{Parameters: }%s\\"
        self.latex_src = template_src % (states_src, params_src)

        self.latex_src = ""
        self.latex_src += r"\begin{eqnarray}"
        for eq in self.equations:
            tmp = latex(self.sympy_dct[eq], mul_symbol="dot")
            self.latex_src += tmp.replace("=", " &=& ") + r"\\"
        self.latex_src += r"\end{eqnarray}"

    def _handle_call_function(self, ins):
        narg = int(ins.arg)

        # hacky way to handle keyword arguments
        if self.kwargs and self.var[-(narg + 1)] == (self.kwargs + ".pop"):
            arg = self.var[-narg][1:-1]
            self.var[-(narg + 1)] = arg
            new_arg = "%s" % arg
            self.signature.append(new_arg)

        else:
            args = [] if narg == 0 else [str(x) for x in self.var[-narg:]]
            func_name = self.var[-(narg + 1)]
            func_globals = get_function_globals(self.model.ode)
            pyfunc = eval(func_name, func_globals)
            sympyfunc = self.pyfunc_to_sympyfunc.get(pyfunc)
            if sympyfunc is not None:
                self.var[-(narg + 1)] = sympyfunc(self, args)
            else:
                self.var[-(narg + 1)] = "{}({})".format(func_name, ",".join(args))

        if narg:
            del self.var[-narg:]

    def _handle_load_attr(self, ins):
        key = ins.argval
        if self.var[-1] == "self":
            if key[:2] == "d_":
                key = key.split("d_")[-1]
                depth = 1
            else:
                depth = 0
            if self.variables[key].integral:
                while self.variables[key].integral is not None:
                    depth += 1
                    key = self.variables[key].integral
            if depth > 0:
                key = "Derivative(%s%s)" % (key, ", t" * depth)
            self.var[-1] = key
        else:
            self.var[-1] += "." + key

    def _handle_store_attr(self, ins):
        self.handle_load_attr(ins)
        rval, lval = self.var[-2:]
        del self.var[-1]
        if rval != lval:
            eqn = "Eq(%s, %s)" % (lval, rval)
            self.equations.append("eqn_%d" % len(self.equations))
            self.var[-1] = "%s = %s" % (self.equations[-1], eqn)
        else:
            del self.var[-1]

    def _handle_store_fast(self, ins):
        key = ins.argval

        prefix, indent = "", ""

        if ins.argval == self.var[-1]:
            del self.var[-1]
            return
        elif self.variables[key].type == "local":

            eqn = "Eq(%s, %s)" % (key, self.var[-1])
            self.equations.append("eqn_%d" % len(self.equations))
            indent = " " * (self.space + self.indent)
            prefix = "with evaluate(False):" + self.newline
            self.var[-1] = "{}{}{} = {}".format(prefix, indent, self.equations[-1], eqn)
        else:
            self.var[-1] = "{} = {}".format(key, self.var[-1])

    def handle_pop_jump_if_true(self, ins):
        self.jump_targets.append(ins.arg)
        self.enter_indent = True
        self.var[-1] = "if not UnevaluatedExpr({0}):".format(self.var[-1])

    def handle_pop_jump_if_false(self, ins):
        self.jump_targets.append(ins.arg)
        self.enter_indent = True
        self.var[-1] = "if UnevaluatedExpr({0}):".format(self.var[-1])

    def handle_jump_forward(self, ins):
        self.leave_indent = True
        self.output_statement()

        target, old_target = ins.argval, self.jump_targets.pop()

        if target != old_target:
            self.var.append("")
            self.enter_indent = True
            self.jump_targets.append(target)
        else:
            self.var.append("")
            self.output_statement()

    def _handle_return_value(self, ins):
        self.var[-1] = ""

    def _py2sympy(*source_funcs):
        def wrapper(func):
            func._source_funcs = source_funcs
            return func

        return wrapper

    def _random_func(func):
        """
        A decorator for registering random functions
        """

        @wraps(func)
        def wrap(self, args):
            self.has_random = True
            return func(self, args)

        return wrap

    def _generate_sympy_func(self, func, args):
        return "%s(%s)" % (func, ", ".join(args))

    @_py2sympy(np.exp)
    def _np_exp(self, args):
        return self._generate_sympy_func("exp", args)

    @_py2sympy(np.log)
    def _np_log(self, args):
        return self._generate_sympy_func("log", args)

    @_py2sympy(np.power)
    def _np_power(self, args):
        return self._generate_sympy_func("pow", args)

    @_py2sympy(np.cbrt)
    def _np_cbrt(self, args):
        return self._generate_sympy_func("cbrt", args)

    @_py2sympy(np.sqrt)
    def _np_sqrt(self, args):
        return self._generate_sympy_func("sqrt", args)

    @_py2sympy(random.gauss, np.random.normal)
    @_random_func
    def _random_gauss(self, args):

        return "N({0}, {1})".format(args[0], args[1])
