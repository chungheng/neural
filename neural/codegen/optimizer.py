"""
Code Generator for optimization of the ode function.
"""
import types

from pycodegen.codegen import CodeGenerator
from pycodegen.utils import get_func_signature

class FuncGenerator(CodeGenerator):
    def __init__(self, model, func, **kwargs):
        self.model = model
        self.func = func
        CodeGenerator.__init__(self, self.func, **kwargs)

        args = get_func_signature(self.func)
        signature = "def {}({}):\n".format(self.func.__name__, ", ".join(args))
        self.ostream.write(signature)

    def handle_load_attr(self, ins):
        key = ins.argval
        if self.var[-1] == 'self':
            if 'd_' in key:
                key = key.split('d_')[-1]
                self.var[-1] += ".gstate['%s']" % key
                return

            for attr in ['states', 'inters', 'params']:
                dct = getattr(self.model, attr)
                if key in dct:
                    self.var[-1] += ".%s['%s']" % (attr, key)
                    return
        self.var[-1] += ".%s" % key
