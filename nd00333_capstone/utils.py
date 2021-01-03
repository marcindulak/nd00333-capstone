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


def get_X_y(df, target_label):
    """
    Split the DataFrame into X and y using target_label as the y label
    """
    return df.drop(labels=[target_label], axis=1), df[target_label]


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