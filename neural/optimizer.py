"""
Code Generator for optimization of the ode function.
"""
import types

from pycodegen.codegen import CodeGenerator
from pycodegen.utils import get_func_signature

class OdeGenerator(CodeGenerator):
    def __init__(self, model, **kwargs):
        self.model = model
        CodeGenerator.__init__(self, model.ode.func_code, **kwargs)

        args = get_func_signature(model.ode)
        self.ostream.write("def ode(%s):\n" % (", ".join(args)))

    def handle_load_attr(self, ins):
        key = ins.argval
        if self.var[-1] == 'self':
            if 'd_' in key:
                key = self.split('d_')[-1]
                self.var[-1] += ".gstate['%s']" % key
                return

            for attr in ['states', 'inters', 'params']:
                dct = getattr(self.model, attr)
                if key in dct:
                    self.var[-1] += ".%s['%s']" % (attr, key)
                    return
        self.var[-1] += ".%s" % key
