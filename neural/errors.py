class NeuralError(Exception):
    """Base Neural Exception Class"""


class NeuralCodeGenError(NeuralError):
    """Base Neural CodeGen Exception"""


class NeuralSymPyCodeGenError(NeuralCodeGenError):
    """Neural SymPy CodeGen Exception"""
