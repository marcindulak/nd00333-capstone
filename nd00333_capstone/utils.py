import logging
from pythonjsonlogger import jsonlogger


def get_logger():
    logger = logging.getLogger()

    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    logger.setLevel(logging.INFO)

    return logger