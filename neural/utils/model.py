# def to_graph(model, local: bool = False):
#     """Convert Circuit to Graph

#     Generate block diagram of the model

#     Parameters:
#         local: Whether to include local variables or not.
#     """
#     try:
#         from .codegen.symbolic import VariableAnalyzer
#     except ImportError as e:
#         raise err.NeuralModelError("'to_graph' requires 'pycodegen'") from e
#     except Exception as e:
#         raise err.NeuralModelError("Unknown Error to 'Model.to_graph' call") from e
#     return VariableAnalyzer(cls).to_graph(local=local)

# def to_latex(model):
#     """Convert Circuit Equation to Latex

#     Generate latex source code for the  model
#     """
#     try:
#         from .codegen.symbolic import SympyGenerator
#     except ImportError as e:
#         raise err.NeuralModelError("'to_latex' requires 'pycodegen'") from e
#     except Exception as e:
#         raise err.NeuralModelError("Unknown Error to 'Model.to_latex' call") from e
#     return SympyGenerator(cls).latex_src
