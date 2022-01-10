"""
Code Generator for optimization of the ode function.
"""
import typing as tp
from inspect import formatargspec, getfullargspec
from pycodegen.codegen import CodeGenerator
from .. import types as tpe

class BaseGenerator(CodeGenerator):
    """Base Code Generator Class"""
    def __init__(self, model: tpe.Model, func: tp.Callable, **kwargs):
        self.model = model
        self.func = func
        CodeGenerator.__init__(self, self.func, **kwargs)
        signature = "def {}{}:\n".format(
            self.func.__name__,
            formatargspec(*getfullargspec(func))
        )
        self.ostream.write(signature)

class FuncGenerator(BaseGenerator):
    def handle_load_attr(self, ins):
        key = ins.argval
        if self.var[-1] == "self":
            if "d_" in key:
                key = key.split("d_")[-1]
                self.var[-1] += ".gstates['%s']" % key
                return

            for attr in ["states", "params"]:
                dct = getattr(self.model, attr)
                if key in dct:
                    self.var[-1] += ".%s['%s']" % (attr, key)
                    return
        self.var[-1] += ".%s" % key


class NumpyGenerator(CodeGenerator):
    def handle_load_attr(self, ins):
        key = ins.argval
        if self.var[-1] == "self":
            if "d_" in key:
                key = key.split("d_")[-1]
                self.var[-1] += ".gstates['%s']" % key
                return

            for attr in ["states", "params"]:
                dct = getattr(self.model, attr)
                if key in dct:
                    self.var[-1] += ".%s['%s']" % (attr, key)
                    return
        self.var[-1] += ".%s" % key

    def handle_store_attr(self, ins):
        """
        symbol1.symbol2 = rvalue
        """
        key = ins.argval
        if self.var[-1] == "self":
            if "d_" in key:
                key = key.split("d_")[-1]
                self.var[-1] += ".gstates['%s'][:]" % key
            else:
                for attr in ["states", "params"]:
                    dct = getattr(self.model, attr)
                    if key in dct:
                        key = ".%s['%s'][:]" % (attr, key)
                        break
        self.var[-1] += ".%s" % key
