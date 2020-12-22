import logging as logger

logger.basicConfig(format="%(asctime)s %(message)s", level=logger.INFO)


class NeuralError(Exception):
    pass


class NeuralUtilityError(NeuralError):
    pass


class SignalTypeNotFoundError(NeuralUtilityError):
    """Raised for when generating an unknown stimulus"""

    pass
