"""
Create AML compute cluster
"""

import argparse
import os

from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException

from nd00333 import utils as package_utils

logger = package_utils.get_logger()


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster-name",
        required=False,
        type=str,
        default=os.environ.get("AML_COMPUTE_CLUSTER_NAME", None),
        help="The name of the cluster. The default is to use the trimmed workspace name",
    )
    parser.add_argument(
        "--cluster-min-nodes",
        required=False,
        type=int,
        default=os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0),
        help="The minimum number of nodes in the cluster",
    )
    parser.add_argument(
        "--cluster-max-nodes",
        required=False,
        type=int,
        default=os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4),
        help="The maximum number of nodes in the cluster",
    )
    parser.add_argument(
        "--cluster-sku",
        required=False,
        type=str,
        default="STANDARD_D2_V3",
        help="The name of the compute instance type",
    )
    parser.add_argument("-f", help="a dummy argument to fool ipython", default=None)

    return parser.parse_args()


def main(args):
    """
    Main
    """

    workspace = package_utils.get_workspace()
    if args.cluster_name is None:
        cluster_name = package_utils.trim_cluster_name(workspace.name)
    else:
        cluster_name = args.cluster_name

    try:
        compute_target = AmlCompute(workspace=workspace, name=cluster_name)
        logger.info(msg=f"Found existing cluster {cluster_name}")
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=args.cluster_sku,
            vm_priority="lowpriority",
            idle_seconds_before_scaledown=40 * 60,
            min_nodes=args.cluster_min_nodes,
            max_nodes=args.cluster_max_nodes,
        )
        compute_target_create = ComputeTarget.create(
            workspace, cluster_name, compute_config
        )

        compute_target_create.wait_for_completion(
            show_output=True, min_node_count=None, timeout_in_minutes=5
        )
        logger.info(
            msg="main", extra={"status": compute_target_create.get_status().serialize()}
        )

    compute_target = workspace.compute_targets[cluster_name]
    logger.info(msg="main", extra={"compute_target": compute_target.serialize()})

    return compute_target


if __name__ == "__main__":
    main(parse_args())
