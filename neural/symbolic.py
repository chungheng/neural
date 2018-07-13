import numpy as np
from StringIO import StringIO
from sympy import *
import types

from pycodegen.codegen import CodeGenerator
from pycodegen.utils import get_func_signature

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

        CodeGenerator.__init__(self, model.ode.func_code)
        _, self.signature, self.kwargs = self._extract_signature(self.model.ode)

        self.variables = {}
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
            self.variables[new_arg] = 'input'
        else:
            args = [] if narg == 0 else self.var[-narg:]
            func_name = self.var[-(narg+1)]
            self.var[-(narg+1)] = func_name + "(%s)" % ','.join(args)

        if narg:
            del self.var[-narg:]

    def handle_load_attr(self, ins):
        self._analyze(ins.arg_name, store=False)
        super(VariableAnalyzer, self).handle_load_attr(ins)

    def handle_store_attr(self, ins):
        self._analyze(ins.arg_name, store=True)
        super(VariableAnalyzer, self).handle_store_attr(ins)

    def handle_store_fast(self, ins):
        self._analyze(ins.arg_name, store=True)
        super(VariableAnalyzer, self).handle_store_fast(ins)

    def _analyze(self, key, store=False):

        if self.var[-1] == 'self':
            if key[:2] == 'd_':
                key = key.split('d_')[-1]
                self.variables[key] = 'state'
            elif store:
                self.variables[key] = 'intermediate'
            elif not store and key not in self.variables:
                self.variables[key] = 'parameter'
        elif store and key not in self.variables:
            self.variables[key] = 'local'

class SympyGenerator(CodeGenerator):
    __metaclass__ = MetaClass
    def __init__(self, model, **kwargs):
        self.model = model
        self.symbol_src = StringIO()
        self.ode_src = StringIO()
        self.states = []
        self.params = []
        self.inputs = []

        self.equations = []

        CodeGenerator.__init__(self, model.ode.func_code, ostream=self.ode_src)

        _, self.signature, self.kwargs = self._extract_signature(self.model.ode)

        self.variables = {}
        # for key in ['Default_States', 'Default_Inters', 'Default_Params']:
        #     if not hasattr(self.model, key):
        #         continue
        #     attr = getattr(self.model, key)
        #     for var in attr.keys():
        #         self.variables[var] = None

        self.generate()
        self.get_symbols()

        self.sympy_dct = {}
        self.latex_src = None

        self.compile_sympy()
        self.generate_latex()

    @property
    def sympy_src(self):
        return self.symbol_src.getvalue() + self.ode_src.getvalue()

    def get_symbols(self):
        self.symbol_src.write("t = Symbol('t')%c" % self.newline)
        for key, val in self.variables.items():
            if val is not None and val != 'local':
                self.symbol_src.write("{0} = Symbol('{0}'){1}".format(key, self.newline))
            if val == 'state':
                self.states.append(key)
            elif val == 'parameter':
                self.params.append(key)

        for key in self.signature:
            self.inputs.append(key)
            self.symbol_src.write("{0} = Symbol('{0}'){1}".format(key, self.newline))

        self.states.sort()
        self.params.sort()
        self.inputs.sort()

    def compile_sympy(self):
        exec(self.sympy_src, globals(), self.sympy_dct)

    def generate_latex(self):
        states_src = ',~'.join([latex(self.sympy_dct[x]) for x in self.states])
        params_src = ',~'.join([latex(self.sympy_dct[x]) for x in self.params])
        template_src = r'\mbox{State Variables: }%s\\\mbox{Parameters: }%s\\'
        self.latex_src = template_src % (states_src, params_src)

        self.latex_src += r'\begin{eqnarray}'
        for eq in self.equations:
            tmp = latex(self.sympy_dct[eq])
            self.latex_src += tmp.replace('=',' &=& ') + r'\\'
        self.latex_src += r'\end{eqnarray}'

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

    def handle_load_attr(self, ins):
        key = ins.arg_name
        if self.var[-1] == 'self':
            if key[:2] == 'd_':
                seg = key.split('d_')
                key = seg[-1]
                self.variables[key] = 'state'
                key = 'Derivative(%s%s)' % (key, ', t'*(len(seg)-1))
            elif self.variables[key] is None:
                self.variables[key] = 'parameter'
            self.var[-1] = key
        else:
            self.var[-1] += ".%s" % key

    def handle_store_attr(self, ins):
        self.handle_load_attr(ins)
        eqn = "Eq(%s, %s)" % (self.var[-1], self.var[-2])
        self.equations.append('eqn_%d' % len(self.equations))
        self.var[-2] = "%s = %s" % (self.equations[-1], eqn)

        del self.var[-1]

    def _handle_store_fast_first_pass(self, ins):

        key = ins.arg_name
        prefix = ''
        if ins.arg_name == self.var[-1]:
            del self.var[-1]
            return
        if (key not in self.variables and key not in self.signature) or \
            self.variables[key] == 'local':
            self.variables[key] = 'local'
        elif key[:2] == 'd_':
            key = key[2:]
            self.variables[key] = 'state'
        else:
            self.variables[key] = 'intermediate'

        self.var[-1] = "%s%s = %s" % (prefix, key, self.var[-1])

    def _handle_store_fast_second_pass(self, ins):

        key = ins.arg_name
        prefix = ''
        if ins.arg_name == self.var[-1]:
            del self.var[-1]
            return
        if (key not in self.variables and key not in self.signature) or \
            self.variables[key] == 'local':
            self.variables[key] = 'local'
            prefix = "with evaluate(False):" + self.newline + " " * (self.indent)
        elif key[:2] == 'd_':
            key = key[2:]
            self.variables[key] = 'state'
            key = 'Derivative(%s)' % key
        else:
            self.variables[key] = 'intermediate'

        self.var[-1] = "%s%s = %s" % (prefix, key, self.var[-1])

    def handle_return_value(self, ins):
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
