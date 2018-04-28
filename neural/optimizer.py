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
        self.modelAttrs = []
        for key in ['Default_States', 'Default_Inters', 'Default_Params']:
            if hasattr(self.model, key):
                self.modelAttrs.append([key, key.split('_')[-1].lower()])

        args = get_func_signature(model.ode)
        self.ostream.write("def ode(%s):\n" % (", ".join(args)))

    def handle_load_attr(self, ins):
        key = ins.arg_name
        if self.var[-1] == 'self':
            for (defaultParam, param)  in self.modelAttrs:
                attr = getattr(self.model, defaultParam)
                if key in attr:
                    self.var[-1] += ".%s['%s']" % (param, key)
                    return
        self.var[-1] += ".%s" % key
