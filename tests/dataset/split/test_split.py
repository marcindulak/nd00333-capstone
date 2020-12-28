import pathlib
import shutil
import inspect
import pandas as pd
import pandas.testing
import pytest

from nd00333_capstone.dataset.split import split
from tests import utils as tests_utils


mock_args = tests_utils.get_mock_args(split.DEFAULT_ARGS)


def test_split_default_args(mocker):
    directory = tests_utils.get_directory(
        "dataset", pathlib.Path(__file__).stem, inspect.currentframe().f_code.co_name
    )
    dataset_path = directory
    dataset_name = "ids2018full"
    mocker.patch(
        "nd00333_capstone.dataset.split.split.parse_args",
        return_value=mock_args(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        ),
    )
    directory_full = pathlib.Path(directory, dataset_name)
    directory_full.mkdir(parents=False, exist_ok=False)
    filename = pathlib.Path(directory_full, "data.csv")
    data = list(zip(range(20), range(20)))  # 20x2 list of tuples
    columns = ["value", "target"]
    _ = pd.DataFrame(data=data, columns=columns).to_csv(filename, index=False)
    args = split.parse_args()
    X_train, X_test, y_train, y_test = split.split(split.parse_args())
    assert X_train.shape == (10, 1)
    assert X_test.shape == (10, 1)
    assert y_train.shape == (10,)
    assert y_test.shape == (10,)
    shutil.rmtree(directory)


def test_split_nondefault_args(mocker):
    directory = tests_utils.get_directory(
        "dataset", pathlib.Path(__file__).stem, inspect.currentframe().f_code.co_name
    )
    dataset_path = directory
    dataset_name = "full"
    dataset_name_train = "training"
    dataset_name_test = "testing"
    sample_fraction = 0.5
    test_size = 0.3
    target_label = "label"
    mocker.patch(
        "nd00333_capstone.dataset.split.split.parse_args",
        return_value=mock_args(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            dataset_name_train=dataset_name_train,
            dataset_name_test=dataset_name_test,
            sample_fraction=sample_fraction,
            test_size=test_size,
            target_label=target_label,
        ),
    )
    directory_full = pathlib.Path(directory, dataset_name)
    directory_full.mkdir(parents=False, exist_ok=False)
    filename = pathlib.Path(directory_full, "data.csv")
    data = list(zip(range(20), range(20)))  # 20x2 list of tuples
    columns = ["value", "label"]
    _ = pd.DataFrame(data=data, columns=columns).to_csv(filename, index=False)
    args = split.parse_args()
    X_train, X_test, y_train, y_test = split.split(split.parse_args())
    assert X_train.shape == (7, 1)
    assert X_test.shape == (3, 1)
    assert y_train.shape == (7,)
    assert y_test.shape == (3,)
    shutil.rmtree(directory)
