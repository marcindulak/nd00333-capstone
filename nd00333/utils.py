import logging
from pythonjsonlogger import jsonlogger

from azureml.core import Workspace


def get_logger():
    """
    Log json
    """
    logger = logging.getLogger()

    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    logger.setLevel(logging.INFO)

    return logger


def get_workspace():
    """
    Get the AzureML workspace from an existing config.json
    """
    workspace = Workspace.from_config()
    return workspace


def get_default_datastore(workspace):
    """
    Get the default datastore of the AzureML workspace
    """
    datastore = workspace.get_default_datastore()
    return datastore


def trim_cluster_name(name):
    """
    It can include letters, digits and dashes. It must start with a letter,
    end with a letter or digit, and be between 2 and 16 characters in length.
    """
    if len(name) <= 16:
        cluster_name = name
    else:
        cluster_name = name[-16:]
    return cluster_name
