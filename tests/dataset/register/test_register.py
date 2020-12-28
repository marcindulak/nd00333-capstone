import collections
import pathlib
import shutil
import inspect
import numpy as np
import pandas as pd
import pandas.testing
import pytest

from nd00333_capstone.dataset.register import register
from tests import utils as tests_utils


mock_datastore = collections.namedtuple("datastore", "name")

ARGS = sorted(register.DEFAULT_ARGS.keys())
mock_args = collections.namedtuple(
    "args",
    (arg[2:].replace("-", "_") for arg in ARGS),
    defaults=(register.DEFAULT_ARGS[arg] for arg in ARGS),
)


@pytest.fixture
def mocker(mocker):
    mocker.patch(
        "nd00333_capstone.dataset.register.register.get_workspace",
        return_value="mocker",
    )
    mocker.patch(
        "nd00333_capstone.dataset.register.register.get_default_datastore",
        return_value=mock_datastore(name="mocker"),
    )
    return mocker


def test_datastore_upload_files_directory_does_not_exist(mocker):
    dataset_path = "mocker"
    dataset_name = "mocker"
    mocker.patch(
        "nd00333_capstone.dataset.register.register.parse_args",
        return_value=mock_args(dataset_path=dataset_path, dataset_name=dataset_name),
    )
    args = register.parse_args()
    directory = pathlib.Path(args.dataset_path, args.dataset_name)
    with pytest.raises(RuntimeError) as excinfo:
        register.datastore_upload_files(register.parse_args())
    assert str(excinfo.value).startswith(
        f"The dataset directory {directory} does not exist"
    )


def test_datastore_upload_files_overwrite_false_succeed(mocker):
    directory = tests_utils.get_directory(
        "dataset", inspect.currentframe().f_code.co_name
    )
    dataset_path = directory.parent
    dataset_name = directory.name
    mocker.patch(
        "nd00333_capstone.dataset.register.register.parse_args",
        return_value=mock_args(
            dataset_path=dataset_path, dataset_name=dataset_name, dataset_version="2"
        ),
    )
    mocker.patch(
        "nd00333_capstone.dataset.register.register.upload_files",
        return_value="",
    )
    filename = pathlib.Path(directory, "1.csv")
    data = [["Benign"]]
    columns = ["Label"]
    _ = pd.DataFrame(data=data, columns=columns).to_csv(filename, index=False)
    args = register.parse_args()
    datastore_path, target_path = register.datastore_upload_files(register.parse_args())
    assert len(datastore_path) == 1
    assert target_path == f"{args.dataset_name}_{args.dataset_version}"