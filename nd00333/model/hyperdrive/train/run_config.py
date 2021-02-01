"""
HyperDrive config
"""

import functools
import operator

from azureml.core import Environment
from azureml.core import ComputeTarget
from azureml.core.dataset import Dataset
from azureml.core.runconfig import RunConfiguration
from azureml.core.script_run_config import ScriptRunConfig

from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import GridParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import choice

#from nd00333.dataset.register import register
#from nd00333.compute import aml_compute
#from nd00333 import utils as package_utils

#logger = package_utils.get_logger()


def get_environment():
    """
    Return an environment
    """
    environment = Environment.from_conda_specification(
        name="hd-train", file_path="./nd00333/model/hyperdrive/train/environment.yml"
    )
    environment.python.user_managed_dependencies = False
    environment.docker.enabled = True
    environment.docker.base_image = (
        "mcr.microsoft.com/azureml/intelmpi2018.3-ubuntu16.04:20200821.v1"
    )
    return environment


def main(
    workspace=package_utils.get_workspace(),
    #dataset_train_name=register.get_default_dataset_name("train"),
    #dataset_validate_name=register.get_default_dataset_name("validate"),
    dataset_train_name="train",
    dataset_validate_name="validate",
):
    """
    Return HyperDriveConfig
    """
    cluster_max_nodes = 8

    compute_target = ComputeTarget(
        workspace=workspace,
        name="nd00333-capstone",
    )
    #args = aml_compute.parse_args()
    #args.cluster_max_nodes = cluster_max_nodes
    #args.cluster_sku = "Standard_D2_v3"
    #compute_target = aml_compute.main(args)
    #logger.info(msg="main", extra={"compute_target": compute_target.serialize()})

    environment = get_environment()
    #logger.info(msg="main", extra={"environment": environment})

    run_config = RunConfiguration()
    run_config.target = compute_target
    run_config.environment = environment
    #logger.info(msg="main", extra={"run_config": run_config})

    parameter_space = {
        "--hyperparameter-n_estimators": choice(range(15, 20 + 1, 5)),
        "--hyperparameter-criterion": choice(["gini", "entropy"]),
        "--hyperparameter-max_depth": choice(range(10, 15 + 1, 5)),
    }
    hyperparameter_sampling = GridParameterSampling(parameter_space)
    hyperparameter_sampling_number_of_runs = functools.reduce(
        operator.mul, [len(value[1][0]) for value in parameter_space.values()]
    )

    train = Dataset.get_by_name(
        workspace=workspace,
        name=dataset_train_name,
    )
    validate = Dataset.get_by_name(
        workspace=workspace,
        name=dataset_validate_name,
    )

    arguments = [
        "--dataset-train-path",
        train.as_named_input("train").as_mount(),
        "--dataset-validate-path",
        validate.as_named_input("validate").as_mount(),
        "--hyperparameter-n_jobs",
        -1,
        "--hyperparameter-random_state",
        0,
    ]

    script_run_config = ScriptRunConfig(
        source_directory="nd00333/model/hyperdrive/train",
        script="train.py",
        arguments=arguments,
        run_config=run_config,
        compute_target=compute_target,
        environment=environment,
        max_run_duration_seconds=60 * 10,
    )

    # The GridParameterSampling is not an iterative process
    # and it won't profit from a termination policy.
    # On the contrary, a highly accurate randomly sampled model may follow an inaccurate model.
    # Therefore a sampling policy that won't terminate any runs is used.
    policy = BanditPolicy(
        evaluation_interval=1, slack_factor=None, slack_amount=1.0, delay_evaluation=0
    )

    hd_config = HyperDriveConfig(
        hyperparameter_sampling=hyperparameter_sampling,
        primary_metric_name="norm_macro_recall",
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=hyperparameter_sampling_number_of_runs,
        max_concurrent_runs=cluster_max_nodes,
        policy=policy,
        run_config=script_run_config,
    )

    return hd_config
