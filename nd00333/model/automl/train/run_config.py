"""
AutoML config
"""

from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig

from nd00333.compute import aml_compute
from nd00333.dataset.register import register

from nd00333 import utils as package_utils

logger = package_utils.get_logger()


def main(
    dataset_trainandvalidate_name=register.get_default_dataset_name("trainandvalidate"),
):
    """
    Return AutoMLConfig
    """
    args = aml_compute.parse_args()
    cluster_max_nodes = 5
    args.cluster_max_nodes = cluster_max_nodes
    args.cluster_sku = "Standard_D12_v2"
    compute_target = aml_compute.main(args)
    logger.info(msg="main", extra={"compute_target": compute_target.serialize()})

    workspace = package_utils.get_workspace()

    trainandvalidate = Dataset.get_by_name(
        workspace=workspace,
        name=dataset_trainandvalidate_name,
    )

    automl_config = AutoMLConfig(
        task="classification",
        iterations=15,
        primary_metric="norm_macro_recall",
        compute_target=compute_target,
        validation_size=0.3,
        featurization="auto",
        max_cores_per_iteration=-1,
        max_concurrent_iterations=cluster_max_nodes,
        allowed_models=["LightGBM", "LogisticRegression", "SGD", "XGBoostClassifier"],
        enable_voting_ensemble=True,
        enable_stack_ensemble=False,
        training_data=trainandvalidate,
        label_column_name="Label",
        experiment_timeout_hours=1.5,
    )

    return automl_config
