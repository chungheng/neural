import logging as logger
logger.basicConfig(
    format='%(asctime)s %(message)s', 
    level=logger.INFO
)


class NeuralError(Exception):
    pass