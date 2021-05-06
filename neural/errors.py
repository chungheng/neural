class NeuralError(Exception):
    """Base Neural Exception Class"""


class NeuralCodeGenError(NeuralError):
    """Base Neural CodeGen Exception"""


class NeuralSymPyCodeGenError(NeuralCodeGenError):
    """Neural SymPy CodeGen Exception"""


class NeuralSymPyCodeGenIdentationError(NeuralSymPyCodeGenError):
    """Neural SymPy CodeGen Indentation Error - Likely caused by elif not understood"""


class NeuralOptimizerError(NeuralError):
    """Base Error for neural.optimize module"""


class NeuralUtilityError(NeuralError):
    """Base Neural Utility Exception"""


class NeuralPlotError(NeuralError):
    """Base Neural Plotting Exception"""
