"""
Deploy a model to AciWebservice
"""

from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice

from nd00333 import utils as package_utils

logger = package_utils.get_logger()


def main(model_name="deploy", model_version=None, deployment_name="deploy"):
    """
    Return a AciWebservice deploy config
    """
    environment = Environment.from_conda_specification(
        name=deployment_name,
        file_path="nd00333/model/deploy/environment.yml",
    )
    logger.info(msg="main", extra={"environment": environment})

    inference_config = InferenceConfig(
        source_directory="nd00333",
        entry_script="model/deploy/score.py",
        base_image="mcr.microsoft.com/azureml/intelmpi2018.3-ubuntu16.04:20200821.v1",
        environment=environment,
    )
    logger.info(msg="main", extra={"inference_config": inference_config})

    workspace = package_utils.get_workspace()

    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1.0,
        memory_gb=8.0,
        auth_enabled=True,
        enable_app_insights=True,
        collect_model_data=False,
    )
    logger.info(msg="main", extra={"deployment_config": deployment_config})

    model = Model(workspace, name=model_name, version=model_version)
    logger.info(msg="main", extra={"model": model})

    service = Model.deploy(
        workspace,
        deployment_name,
        [model],
        inference_config,
        deployment_config,
        overwrite=True,
    )
    logger.info(msg="main", extra={"service": service})

    return service
