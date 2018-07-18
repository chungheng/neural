import numpy as np
import os
from StringIO import StringIO
from sympy import *
import types

from pycodegen.codegen import CodeGenerator
from pycodegen.utils import get_func_signature

class _Variable(object):
    default = {
        'type': None,
        'integral': None,
        'derivative': None
    }
    def __init__(self, **kwargs):
        for key, val in self.default.items():
            val = kwargs.pop(key, val)
            self.__dict__[key] = val
        if len(kwargs):
            raise AttributeError('Invalid attribute: %s' % kwargs.keys()[0])
    def __setattribute__(self, key, val):
        if key not in self.__dict__:
            raise KeyError('Unrecognized key: %s' % key)
        self.__dict__[key] = val

class MetaClass(type):
    def __new__(cls, clsname, bases, dct):
        py2sympy = dict()
        for key, val in dct.items():
            if callable(val) and hasattr(val, '_source_funcs'):
                for func in val._source_funcs:
                    py2sympy[func] = val
        dct['pyfunc_to_sympyfunc'] = py2sympy
        return super(MetaClass, cls).__new__(cls, clsname, bases, dct)


class VariableAnalyzer(CodeGenerator):
    """
    Analyze the variables in a set of ODEs
    """

    def __init__(self, model, **kwargs):
        self.model = model
        self.equations = []

        _, self.signature, self.kwargs = self._extract_signature(self.model.ode)
        with open(os.devnull, 'w') as f:
            CodeGenerator.__init__(self, model.ode.func_code, ostream=f)
            self.variables = {}
            self.generate()
            self.generate()

    def _extract_signature(self, func):
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

    def handle_call_function(self, ins):
        narg = int(ins.arg)

        # hacky way to handle keyword arguments
        if self.kwargs and self.var[-(narg+1)] == (self.kwargs + ".pop"):
            arg = self.var[-narg][1:-1]
            self.var[-(narg+1)] = arg
            new_arg = "%s" % arg
            self.signature.append(new_arg)
            self.variables[new_arg] = _Variable(type='input')
        else:
            args = [] if narg == 0 else self.var[-narg:]
            func_name = self.var[-(narg+1)]
            self.var[-(narg+1)] = func_name + "(%s)" % ','.join(args)

        if narg:
            del self.var[-narg:]

    def handle_store_fast(self, ins):
        """
        symbol1 = rvalue
        """
        key = ins.arg_name
        if key not in self.variables:
            self.variables[key] = _Variable(type='local')
        self.var[-1] = key + ' = ' + self.var[-1]

    def handle_load_attr(self, ins):
        """
        ... symbol1.symbol2 ...
        """
        key = ins.arg_name
        if self.var[-1] == 'self':
            if key[:2] == 'd_':
                key = key.split('d_')[-1]
                self._set_variable(key, type='state')
            elif key not in self.variables:
                self._set_variable(key, type='parameter')
            self.var[-1] = key
        else:
            self.var[-1] += "." + ins.arg_name

    def handle_store_attr(self, ins):
        """
        symbol1.symbol2 = rvalue
        """
        key = ins.arg_name
        if self.var[-1] == 'self':
            if key[:2] == 'd_':
                key = key.split('d_')[-1]
                self._set_variable(key, type='state')
                if self.var[-2] in self.variables:
                    self.variables[key].derivative = self.var[-2]
                    self.variables[self.var[-2]].integral = key
            elif key not in self.variables:
                self._set_variable(key, type='intermediate')
            self.var[-1] = key
        else:
            self.var[-1] += "." + ins.arg_name
        self.var[-2] = self.var[-1] + ' = ' + self.var[-2]
        del self.var[-1]

    def _set_variable(self, variable, **kwargs):
        if variable not in self.variables:
            self.variables[variable] = _Variable()

        for key, val in kwargs.items():
            setattr(self.variables[variable], key, val)


class SympyGenerator(VariableAnalyzer):
    __metaclass__ = MetaClass
    def __init__(self, model, **kwargs):
        VariableAnalyzer.__init__(self, model, **kwargs)
        # for attr in ['state', 'parameter', 'input']:
        #     lst = [k for k, v in self.variables.items() if v.type == attr]
        #     lst.sort()
        #     setattr(self, attr+'s', lst)
        self.ode_src = StringIO()
        self.symbol_src = StringIO()
        self.ostream = self.ode_src

        for key in dir(self):
            if key[:8] == '_handle_':
                setattr(self, key[1:], getattr(self, key))
        #
        self.generate()
        # print self.ode_src.getvalue()
        self.get_symbols()

        self.sympy_dct = {}
        self.latex_src = None
        #
        self.compile_sympy()
        self.generate_latex()


    @property
    def sympy_src(self):
        return self.symbol_src.getvalue() + self.ode_src.getvalue()

    def get_symbols(self):
        self.symbol_src.write("t = Symbol('t')%c" % self.newline)
        for key, val in self.variables.items():
            if val.type != 'local':
                src = "{0} = Symbol('{0}'){1}".format(key, self.newline)
                self.symbol_src.write(src)

    def compile_sympy(self):
        exec(self.sympy_src, globals(), self.sympy_dct)

    def generate_latex(self):
        cond = lambda v: v.type == 'state' and v.integral is None
        states = [k for k, v in self.variables.items() if cond(v)]
        states.sort()
        states_src = ',~'.join([latex(self.sympy_dct[x]) for x in states])
        params = [k for k, v in self.variables.items() if v.type == 'parameter']
        params.sort()
        params_src = ',~'.join([latex(self.sympy_dct[x]) for x in params])
        template_src = r'\mbox{State Variables: }%s\\\mbox{Parameters: }%s\\'
        self.latex_src = template_src % (states_src, params_src)

        self.latex_src += r'\begin{eqnarray}'
        for eq in self.equations:
            tmp = latex(self.sympy_dct[eq])
            self.latex_src += tmp.replace('=',' &=& ') + r'\\'
        self.latex_src += r'\end{eqnarray}'

    def _handle_call_function(self, ins):
        narg = int(ins.arg)

        # hacky way to handle keyword arguments
        if self.kwargs and self.var[-(narg+1)] == (self.kwargs + ".pop"):
            arg = self.var[-narg][1:-1]
            self.var[-(narg+1)] = arg
            new_arg = "%s" % arg
            self.signature.append(new_arg)

        else:
            args = [] if narg == 0 else self.var[-narg:]
            func_name = self.var[-(narg+1)]
            pyfunc = eval(func_name, self.model.ode.func_globals)
            sympyfunc = self.pyfunc_to_sympyfunc.get(pyfunc)
            if sympyfunc is not None:
                self.var[-(narg+1)] = sympyfunc(self, args)
            else:
                self.var[-(narg+1)] = func_name + "(%s)" % ','.join(args)

        if narg:
            del self.var[-narg:]

    def _handle_load_attr(self, ins):
        key = ins.arg_name
        if self.var[-1] == 'self':
            if key[:2] == 'd_':
                seg = key.split('d_')
                key = 'Derivative(%s%s)' % (seg[-1], ', t'*(len(seg)-1))
            self.var[-1] = key
        else:
            self.var[-1] += "." + key

    def _handle_store_attr(self, ins):
        self.handle_load_attr(ins)
        rval, lval = self.var[-2:]
        del self.var[-1]
        cond = rval in self.variables and self.variables[rval].type == 'state'
        if not (cond and 'Derivative' in lval):
            eqn = "Eq(%s, %s)" % (lval, rval)
            self.equations.append('eqn_%d' % len(self.equations))
            self.var[-1] = "%s = %s" % (self.equations[-1], eqn)
        else:
            del self.var[-1]


    def _handle_store_fast(self, ins):
        key = ins.arg_name

        prefix = ''
        if ins.arg_name == self.var[-1]:
            del self.var[-1]
            return
        elif self.variables[key].type == 'local':
            prefix = "with evaluate(False):" + self.newline + " "*self.indent

        self.var[-1] = "%s%s = %s" % (prefix, key, self.var[-1])

    def _handle_return_value(self, ins):
        self.var[-1] = ""

    def _py2sympy(*source_funcs):
        def wrapper(func):
            func._source_funcs = source_funcs
            return func
        return wrapper

    def _generate_sympy_func(self, func, args):
        return "%s(%s)" % (func, ', '.join(args))


    @_py2sympy(np.exp)
    def _np_exp(self, args):
        return self._generate_sympy_func('exp', args)

    @_py2sympy(np.power)
    def _np_power(self, args):
        return self._generate_sympy_func('pow', args)

    @_py2sympy(np.cbrt)
    def _np_cbrt(self, args):
        return self._generate_sympy_func('cbrt', args)
