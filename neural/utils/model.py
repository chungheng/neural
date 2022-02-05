import struct
import zlib
from binascii import unhexlify
from .. import types as tpe

def chunk(tipe, data):
    return (
        struct.pack(">I", len(data))
        + tipe
        + data
        + struct.pack(">I", zlib.crc32(tipe + data))
    )


MINIMUM_PNG = (
    b"\x89PNG\r\n\x1A\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
    + chunk(b"IDAT", unhexlify(b"789c6300010000050001"))
    + chunk(b"IEND", b"")
)


def to_graph(model: tpe.Model, local: bool = False):
    """Convert Circuit to Graph

    Generate block diagram of the model

    Parameters:
        local: Whether to include local variables or not.
    """
    raise NotImplementedError
    # try:
    #     from .codegen.symbolic import VariableAnalyzer
    # except ImportError as e:
    #     raise err.NeuralModelError("'to_graph' requires 'pycodegen'") from e
    # except Exception as e:
    #     raise err.NeuralModelError("Unknown Error to 'Model.to_graph' call") from e
    # return VariableAnalyzer(cls).to_graph(local=local)

def to_latex(model: tpe.Model):
    """Convert Circuit Equation to Latex

    Generate latex source code for the  model
    """
    raise NotImplementedError
    # try:
    #     from .codegen.symbolic import SympyGenerator
    # except ImportError as e:
    #     raise err.NeuralModelError("'to_latex' requires 'pycodegen'") from e
    # except Exception as e:
    #     raise err.NeuralModelError("Unknown Error to 'Model.to_latex' call") from e
    # return SympyGenerator(cls).latex_src