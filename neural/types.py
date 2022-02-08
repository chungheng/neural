import typing as tp
import sympy as sp
import numpy.typing as npt
from numbers import Number

Network = "neural.network.Network"
Input = "neural.network.Input"
Container = "neural.network.Container"
Symbol = "neural.network.Symbol"
Operator = "neural.baseoperator.Operator"
Solver = tp.Union[str, "neural.solver.BaseSolver"]
Model = "neural.basemodel.Model"
SupportedBackend = tp.Literal["scalar", "numpy", "cuda"]
ScalarOrArray = tp.Union[Number, npt.ArrayLike]
ModelAttrType = tp.Literal["params", "states", "gstates"]
ModelSymbol = tp.Union[sp.Symbol, sp.Function, sp.Derivative]
