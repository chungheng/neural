class NeuralError(Exception):
    """Base Neural Exception Class"""


class NeuralCodeGenError(NeuralError):
    """Base Neural CodeGen Exception"""


class NeuralSymPyCodeGenError(NeuralCodeGenError):
    """Neural SymPy CodeGen Exception"""


class NeuralBackendError(NeuralError):
    """Neural Backend Module Exception"""


class NeuralSymPyCodeGenIdentationError(NeuralSymPyCodeGenError):
    """Neural SymPy CodeGen Indentation Error - Likely caused by elif not understood"""


class NeuralOptimizerError(NeuralError):
    """Base Error for neural.optimize module"""


class NeuralUtilityError(NeuralError):
    """Base Neural Utility Exception"""


class NeuralPlotError(NeuralError):
    """Base Neural Plotting Exception"""

class NeuralWarning(Warning):
    """Base Neural Warning Class"""


class NeuralError(Exception):
    """Base Neural Exception Class"""


class NeuralModelError(NeuralError):
    """Neural Exception for Model Module"""


class NeuralModelWarning(NeuralWarning):
    """Neural Warning for Model Module"""


class NeuralRecorderError(NeuralError):
    """Neural Exception for Recorder Module"""


class NeuralRecorderWarning(NeuralWarning):
    """Neural Warning for Recorder Module"""


class NeuralBackendError(NeuralError):
    """Backend Exception"""


class NeuralBackendWarning(NeuralWarning):
    """Backend Warning"""


class NeuralNetworkError(NeuralError):
    """Neural Error for Network Module"""


class NeuralContainerError(NeuralNetworkError):
    """Neural Error for Container Module used in Network constructions"""


class NeuralNetworkCompileError(NeuralNetworkError):
    """Neural Network Compilation Error"""


class NeuralNetworkInputError(NeuralNetworkError):
    """Neural Network Input Object Error"""


class NeuralNetworkUpdateError(NeuralNetworkError):
    """Container's Update Faield in Neural Network"""


class NeuralUtilityError(NeuralError):
    """Neural Error for Utility Functions

    Includes:

        1. `utils.py`
        2. `plot.py`
    """


class SignalTypeNotFoundError(NeuralUtilityError):
    """Raised for when generating an unknown stimulus"""


class NeuralUtilityWarning(NeuralWarning):
    """Neural Error for Utility Functions

    Includes:

        1. `utils.py`
        2. `plot.py`
    """


class NeuralNetworkWarning(NeuralWarning):
    """Neural Error for Network Module unctions"""


class NeuralCodeGenError(NeuralError):
    """Base Neural CodeGen Exception"""


class NeuralSymPyCodeGenError(NeuralCodeGenError):
    """Neural SymPy CodeGen Exception"""


class NeuralSymPyCodeGenIdentationError(NeuralSymPyCodeGenError):
    """Neural SymPy CodeGen Indentation Error - Likely caused by elif not understood"""


class NeuralOptimizerError(NeuralError):
    """Base Error for neural.optimize module"""


class NeuralPlotError(NeuralError):
    """Base Neural Plotting Exception"""
