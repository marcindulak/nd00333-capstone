"""
Common package utilities
"""

import json
import logging
import os

try:
    from pythonjsonlogger import jsonlogger
except ModuleNotFoundError as exc:
    print(f"{exc.msg}: using the default formatter as the fallback")
    jsonlogger = None

from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model
from azureml.train.automl.run import AutoMLRun
from azureml.train.hyperdrive.run import HyperDriveRun


def get_logger():
    """
    Log json or fallback to the default logging.Formatter
    """
    logger = logging.getLogger()

    handler = logging.StreamHandler()
    if jsonlogger:
        formatter = jsonlogger.JsonFormatter()
    else:
        formatter = logging.Formatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


def get_sp_auth():
    """
    Get authentication object of the service principal or None
    """
    input_azure_credentials = os.environ.get("INPUT_AZURE_CREDENTIALS", default="{}")
    if input_azure_credentials == "{}":
        return None
    try:
        azure_credentials = json.loads(input_azure_credentials)
    except json.JSONDecodeError:
        raise RuntimeError("Invalid INPUT_AZURE_CREDENTIALS")

    return ServicePrincipalAuthentication(
        tenant_id=azure_credentials.get("tenantId", ""),
        service_principal_id=azure_credentials.get("clientId", ""),
        service_principal_password=azure_credentials.get("clientSecret", ""),
        cloud="AzureCloud",
    )


def get_workspace():
    """
    Get the AzureML workspace from an existing config.json
    """
    auth = get_sp_auth()
    if auth:
        workspace = Workspace.from_config(auth=auth)
    else:
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
    Return AzureML compatible cluster name.
    It can include letters, digits and dashes. It must start with a letter,
    end with a letter or digit, and be between 2 and 16 characters in length.
    """
    if len(name) <= 16:
        cluster_name = name
    else:
        cluster_name = name[-16:]
    return cluster_name


def get_best_run(experiment, run):
    """
    Return the best among child runs
    """
    best_run = None
    if run.type == "automl":
        get_run = AutoMLRun(experiment=experiment, run_id=run.id)
        best_run = get_run.get_best_child()
    if run.type == "hyperdrive":
        get_run = HyperDriveRun(experiment=experiment, run_id=run.id)
        best_run = get_run.get_best_run_by_primary_metric()

    return best_run


def register_model(
    model_name,
    model_path="outputs/model.pkl",
    run=None,
):
    """
    Register model into the current workspace.
    In case of run.register_model model_path is remote,
    but in case of Model.register model_path is local.
    Note the inconsistent order of arguments in Run.register_model and Model.register.
    """
    if run:
        model = run.register_model(
            model_name=model_name,
            model_path=model_path,
        )
    else:
        workspace = get_workspace()
        model = Model.register(
            workspace=workspace,
            model_path=model_path,
            model_name=model_name,
        )
    return model


def service_predict_batch(service, dataframe, batch_size=1, verbose=0):
    """
    Return predictions for the dataframe from a service by making batch requests
    """
    predictions = []
    batch_number_total = dataframe.shape[0] // batch_size
    for batch_number, batch_start in enumerate(
        range(0, dataframe.shape[0], batch_size)
    ):
        if verbose and batch_number % 100 == 0:
            print(f"batch number {batch_number} of {batch_number_total}")
        batch = dataframe.iloc[batch_start : batch_start + batch_size, :].to_dict(
            orient="records"
        )
        batch_json = json.dumps({"data": batch})
        result = json.loads(service.run(batch_json))["result"]
        predictions.extend(result)
    return predictions
