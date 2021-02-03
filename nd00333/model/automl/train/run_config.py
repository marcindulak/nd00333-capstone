"""
AutoML config
"""

from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig

from nd00333.compute import aml_compute
from nd00333.dataset.register import config

from nd00333 import utils as package_utils

logger = package_utils.get_logger()


def main(
    workspace=None,
    dataset_trainandvalidate_name=config.get_default_dataset_name("trainandvalidate"),
):
    """
    Return AutoMLConfig
    """

    if not workspace:
        workspace = package_utils.get_workspace()

    args = aml_compute.parse_args()
    cluster_max_nodes = 5
    args.cluster_max_nodes = cluster_max_nodes
    args.cluster_sku = "Standard_D12_v2"
    compute_target = aml_compute.main(args)
    logger.info(msg="main", extra={"compute_target": compute_target.serialize()})

    trainandvalidate = Dataset.get_by_name(
        workspace=workspace,
        name=dataset_trainandvalidate_name,
    )

    model_settings = {
        "task": "classification",
        "primary_metric": "norm_macro_recall",
    }

    ensemble_settings = {
        "iterations": 15,
        "allowed_models": ["LightGBM", "LogisticRegression", "SGD", "XGBoostClassifier"],
        "enable_voting_ensemble": True,
        "enable_stack_ensemble": False,
    }

    dataset_settings = {
        "validation_size": 0.3,
        "featurization": "auto",
        "training_data": trainandvalidate,
        "label_column_name": "Label",
    }

    compute_settings = {
        "compute_target": compute_target,
        "max_cores_per_iteration": -1,
        "max_concurrent_iterations": cluster_max_nodes,
        "experiment_timeout_hours": 1.5,
    }

    automl_config = AutoMLConfig(
        **model_settings,
        **ensemble_settings,
        **dataset_settings,
        **compute_settings,
    )

    return automl_config
