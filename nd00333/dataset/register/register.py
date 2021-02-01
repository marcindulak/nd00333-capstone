"""
Register a dataset in AzureML
"""

import argparse
import glob
import os
import pathlib


from azureml.core.dataset import Dataset
from azureml.data import DataType
from azureml.data.datapath import DataPath
from azureml.data.dataset_factory import TabularDatasetFactory

from nd00333.dataset.load import load
from nd00333 import utils as package_utils

logger = package_utils.get_logger()

DATASET_NAME = "ids2018"
DATASET_VERSION = "1"

DTYPE_MAP = {"int64": DataType.to_long(), "object": DataType.to_string()}
DATA_TYPES = {feature: DTYPE_MAP[dtype] for feature, dtype in load.DTYPE.items()}
DEFAULT_ARGS = {
    "--dataset-path": "datasets",
    "--dataset-name": DATASET_NAME,
    "--dataset-version": DATASET_VERSION,
    "--dataset-type": "file",
    "--dataset-overwrite": False,
    "--dry-run": False,
}


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        required=False,
        type=str,
        default=DEFAULT_ARGS["--dataset-path"],
        help="Root path of the local datasets",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        type=str,
        default=DEFAULT_ARGS["--dataset-name"],
        help="Name of the input dataset",
    )
    parser.add_argument(
        "--dataset-version",
        required=False,
        type=str,
        default=DEFAULT_ARGS["--dataset-version"],
        help="Version of the dataset",
    )
    parser.add_argument(
        "--dataset-type",
        required=False,
        type=str,
        default=DEFAULT_ARGS["--dataset-type"],
        choices=["file", "tabular"],
        help="The type AzureML used to load the input dataset",
    )
    parser.add_argument(
        "--dataset-overwrite",
        required=False,
        action="store_true",
        default=DEFAULT_ARGS["--dataset-overwrite"],
        help="Whether to overwrite the remote dataset",
    )
    parser.add_argument(
        "--dry-run",
        required=False,
        action="store_true",
        default=DEFAULT_ARGS["--dry-run"],
        help="Dry run",
    )
    return parser.parse_args()


def upload_files(datastore, **kwargs):
    """
    Upload files to the datastore
    """
    return datastore.upload_files(**kwargs)


def datastore_upload_files(args):
    """
    Get the default datastore and upload files into it
    """
    workspace = package_utils.get_workspace()
    datastore = package_utils.get_default_datastore(workspace)

    directory = pathlib.Path(args.dataset_path, args.dataset_name)
    if not os.path.exists(directory):
        msg = f"The dataset directory {directory} does not exist"
        logger.exception(msg)
        raise RuntimeError(msg)

    files = [os.path.abspath(file) for file in sorted(glob.glob(f"{directory}/*.csv"))]
    target_path = f"{args.dataset_name}_{args.dataset_version}"
    kwargs = {
        "files": files,
        "target_path": target_path,
        "overwrite": args.dataset_overwrite,
    }
    logger.info(msg="datastore.upload_files", extra={"kwargs": kwargs})
    if not args.dry_run:
        try:
            _ = upload_files(datastore, **kwargs)
        except:
            msg = f"Upload to target_path {target_path} failed"
            logger.exception(msg)
            raise RuntimeError(msg)

    datastore_path = [
        DataPath(datastore, str(pathlib.Path(target_path, os.path.basename(file))))
        for file in files
    ]

    return datastore_path, target_path


def dataset_register_tabular(args):
    """
    Register a tabular dataset into the workspace
    """
    workspace = package_utils.get_workspace()

    datastore_path, target_path = datastore_upload_files(args)

    kwargs = {"path": datastore_path, "set_column_types": DATA_TYPES}
    logger.info(
        msg="TabularDatasetFactory.from_delimited_files", extra={"kwargs": kwargs}
    )
    if not args.dry_run:
        tabular = TabularDatasetFactory.from_delimited_files(**kwargs)

    kwargs = {"workspace": workspace, "name": target_path, "create_new_version": False}
    logger.info(msg="tabular.register", extra={"kwargs": kwargs})
    if not args.dry_run:
        _ = tabular.register(**kwargs)


def dataset_register_file(args):
    """
    Register a file dataset into the workspace
    """
    workspace = package_utils.get_workspace()

    datastore_path, target_path = datastore_upload_files(args)

    logger.info(msg="Dataset.File.from_files", extra={"datastore_path": datastore_path})
    if not args.dry_run:
        file_dataset = Dataset.File.from_files(path=datastore_path)

    kwargs = {"workspace": workspace, "name": target_path, "create_new_version": False}
    logger.info(msg="file_dataset.register", extra={"kwargs": kwargs})
    if not args.dry_run:
        _ = file_dataset.register(**kwargs)


def dataset_register(args):
    """
    Register either a file or tabular dataset in AzureML
    """
    if args.dataset_type == "file":
        dataset_register_file(args)
    elif args.dataset_type == "tabular":
        dataset_register_tabular(args)
    else:
        msg = f"The dataset type {args.dataset_type} is not supported"
        logger.exception(msg)
        raise RuntimeError(msg)


def get_default_dataset_name(dataset_type):
    """
    Return the default dataset name
    """
    return f"{DATASET_NAME}{dataset_type}{DATASET_VERSION}"


if __name__ == "__main__":
    dataset_register(parse_args())
