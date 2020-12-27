class NeuralWarning(Warning):
    """Base Neural Warning Class"""


class NeuralError(Exception):
    """Base Neural Exception Class"""


class NeuralBackendError(NeuralError):
    """Backend Exception"""


class NeuralBackendWarning(NeuralWarning):
    """Backend Warning"""


class NeuralNetworkError(NeuralError):
    """Neural Error for Network Module"""


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
