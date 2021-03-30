class NeuralError(Exception):
    """Base Neural Exception Class"""


class NeuralCodeGenError(NeuralError):
    """Base Neural CodeGen Exception"""


class NeuralSymPyCodeGenError(NeuralCodeGenError):
    """Neural SymPy CodeGen Exception"""


class NeuralOptimizerError(NeuralError):
    """Base Error for neural.optimize module"""


class NeuralUtilityError(NeuralError):
    """Base Neural Utility Exception"""


class NeuralPlotError(NeuralError):
    """Base Neural Plotting Exception"""
