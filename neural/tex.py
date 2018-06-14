
from pycodegen.codegen import CodeGenerator
from pycodegen.utils import get_func_signature


class MetaClass(type):
    def __new__(cls, clsname, bases, dct):
        py2tex = dict()
        for key, val in dct.items():
            if callable(val) and hasattr(val, '_source_funcs'):
                for func in val._source_funcs:
                    py2tex[func] = val
        dct['pyfunc_to_tex'] = py2tex
        return super(MetaClass, cls).__new__(cls, clsname, bases, dct)

class CudaGenerator(CodeGenerator):
    __metaclass__ = MetaClass

    def __init__(self, model, **kwargs):
        self.dtype = dtype_to_ctype(kwargs.pop('dtype', np.float32))
        self.model = model
        self.params_gdata = kwargs.pop('params_gdata', [])
        self.inputs_gdata = kwargs.pop('inputs_gdata', dict())

        self.ode_src = StringIO()
        self.post_src = StringIO()
        cls = self.model.__class__
        self.has_post = np.all([cls.post != base.post for base in cls.__bases__])

        CodeGenerator.__init__(self, model.ode.func_code, newline=';\n',
                offset=4, ostream=self.ode_src, **kwargs)

        self.tpl = Template(cuda_src_template)

    def generate(self, instructions=None):
        if self.has_post and not len(self.post_src.getvalue()):
            self.generate_post()
        if not len(self.ode_src.getvalue()):
            self.generate_ode()

    def generate_ode(self):
        pass

    def generate_post(self):
        pass
