"""
Shared config for the datasets registered in AzureML
"""

DATASET_NAME = "ids2018"
DATASET_VERSION = "1"


def get_default_dataset_name(dataset_type):
    """
    Return the default dataset name
    """
    return f"{DATASET_NAME}{dataset_type}{DATASET_VERSION}"
