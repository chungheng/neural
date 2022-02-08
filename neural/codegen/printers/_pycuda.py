from sympy.printing.c import C99CodePrinter


class PyCUDAPrinter(C99CodePrinter):
    def __init__(self, parsed_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = parsed_model

    def _print_Relational(self, expr):
        """Convert sympy.Eq to Assignments"""
        if expr in self.model.ode and isinstance(expr, sp.Eq):
            lhs_code = self._print(expr.lhs)
            rhs_code = self._print(expr.rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))
        return super()._print_Relational(expr)

    def _print_Symbol(self, expr):
        for var in ["params", "internals", "inputs"]:
            sym_to_name = {
                val.sym: val.name for val in getattr(self.model, var).values()
            }
            if expr in sym_to_name:
                if var in ["internals", "inputs"]:
                    return sym_to_name[expr]
                else:
                    return f"{var}.{sym_to_name[expr]}"
        return super()._print_Symbol(expr)

    def _print_Function(self, expr):
        for var in ["states", "gstates"]:
            sym_to_name = {
                val.sym: val.name for val in getattr(self.model, var).values()
            }
            if expr in sym_to_name:
                return f"{var}.{sym_to_name[expr]}"
        return super()._print_Function(expr)

    def _print_Derivative(self, expr):
        for var in ["gstates"]:
            sym_to_name = {
                val.sym: val.name for val in getattr(self.model, var).values()
            }
            if expr in sym_to_name:
                return f"{var}.{sym_to_name[expr]}"
        return super()._print_Function(expr)
