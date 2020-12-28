import pathlib
import shutil
import inspect
import numpy as np
import pandas as pd
import pandas.testing
import pytest

from nd00333_capstone.dataset.load import load
from tests import utils as tests_utils

LABEL_FRACTIONS = {
    "Benign": 0.842494,
    "Bot": 0.017646,
    "Brute Force -Web": 3.3e-05,
    "Brute Force -XSS": 1.4e-05,
    "DDOS attack-HOIC": 0.020204,
    "DDoS attacks-LOIC-HTTP": 0.035694,
    "DoS attacks-GoldenEye": 0.003427,
    "DoS attacks-Hulk": 0.003185,
    "DoS attacks-SlowHTTPTest": 0.01726,
    "DoS attacks-Slowloris": 0.001065,
    "FTP-BruteForce": 0.023857,
    "Infilteration": 0.01197,
    "SQL Injection": 7e-06,
    "SSH-Bruteforce": 0.023145,
}


def test_get_df_from_csv_default():
    directory = tests_utils.get_directory(
        "dataset", pathlib.Path(__file__).stem, inspect.currentframe().f_code.co_name
    )
    filename = pathlib.Path(directory, "data.csv")
    data = [["Benign"]]
    columns = ["Label"]
    _ = pd.DataFrame(data=data, columns=columns).to_csv(filename, index=False)
    df = load.get_df_from_csv(filename)
    assert list(df.columns) == ["Label"]
    assert list(df.dtypes) == ["object"]
    shutil.rmtree(directory)


def test_get_df_from_csv_dtype_int():
    directory = tests_utils.get_directory(
        "dataset", pathlib.Path(__file__).stem, inspect.currentframe().f_code.co_name
    )
    filename = pathlib.Path(directory, "data.csv")
    data = ["123", "456"]
    dtype = [("Label", np.dtype(int))]
    records = np.array(data, dtype=dtype)
    _ = pd.DataFrame.from_records(records).to_csv(filename, index=False)
    df = load.get_df_from_csv(filename)
    assert list(df.columns) == ["Label"]
    assert list(df.dtypes) == ["int"]
    shutil.rmtree(directory)


def test_get_df_from_csv_usecols():
    directory = tests_utils.get_directory(
        "dataset", pathlib.Path(__file__).stem, inspect.currentframe().f_code.co_name
    )
    filename = pathlib.Path(directory, "data.csv")
    data = [("1", "Benign"), ("2", "Malicious")]
    dtype = [("dummy1", np.dtype(int)), ("Label", np.dtype(object))]
    records = np.array(data, dtype=dtype)
    usecols = ["Label"]
    _ = pd.DataFrame.from_records(records).to_csv(filename, index=False)
    df = load.get_df_from_csv(filename, usecols=usecols)
    assert list(df.columns) == ["Label"]
    assert list(df.dtypes) == ["object"]
    shutil.rmtree(directory)


def test_get_df_from_directory_default():
    directory = tests_utils.get_directory(
        "dataset", pathlib.Path(__file__).stem, inspect.currentframe().f_code.co_name
    )
    columns = ["Label"]
    for iter in [0, 1, 2]:
        _ = pd.DataFrame(data=[iter], columns=columns).to_csv(
            pathlib.Path(directory, str(iter) + ".csv"), index=False
        )
    df = load.get_df_from_directory(directory)
    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame(
            data=[[0], [1], [2]],
            columns=columns,
        ),
    )
    shutil.rmtree(directory)


@pytest.fixture(scope="session")
def dataset():
    def inner(name):
        directory = pathlib.Path(
            "datasets",
            name,
        )
        if directory.exists():
            return load.get_df_from_directory(directory)
        else:
            pytest.skip(f"The dataset directory {directory} does not exist")

    return inner


def test_get_df_from_directory_dataset_full_columns(dataset):
    assert list(sorted(dataset("ids2018full").columns)) == sorted(load.DTYPE.keys())


def test_get_df_from_directory_dataset_full_dtype(dataset):
    columns = load.DTYPE.keys()
    assert list(dataset("ids2018full").dtypes[columns]) == [
        load.DTYPE[column] for column in columns
    ]


def test_get_df_from_directory_dataset_full_labels(dataset):
    grouped = (
        dataset("ids2018full")[["Flow Duration", "Label"]]
        .groupby("Label")
        .agg(count=("Flow Duration", "count"))
    )
    assert list(sorted(grouped.index)) == sorted(load.LABEL_TO_INDEX.keys())


def test_get_df_from_directory_dataset_full_fractions(dataset):
    grouped = (
        dataset("ids2018full")[["Flow Duration", "Label"]]
        .groupby("Label")
        .agg(count=("Flow Duration", "count"))
    )
    reference = (
        pd.DataFrame.from_dict(LABEL_FRACTIONS, orient="index")
        .reset_index()
        .rename(columns={"index": "Label", 0: "count"})
    ).set_index("Label")
    result = (
        (grouped["count"] / grouped["count"].sum()).reset_index().set_index("Label")
    )
    for label in sorted(LABEL_FRACTIONS.keys()):
        assert np.isclose(result.loc[label], reference.loc[label], rtol=0.1, atol=0)


def test_get_df_from_directory_dataset_train_fractions(dataset):
    grouped = (
        dataset("ids2018train")[["Flow Duration", "Label"]]
        .groupby("Label")
        .agg(count=("Flow Duration", "count"))
    )
    reference = (
        pd.DataFrame.from_dict(LABEL_FRACTIONS, orient="index")
        .reset_index()
        .rename(columns={"index": "Label", 0: "count"})
    ).set_index("Label")
    result = (
        (grouped["count"] / grouped["count"].sum()).reset_index().set_index("Label")
    )
    for label in sorted(LABEL_FRACTIONS.keys()):
        if label in ["Brute Force -XSS", "SQL Injection"]:
            rtol = 0.3
        else:
            rtol = 0.1
        assert np.isclose(result.loc[label], reference.loc[label], rtol=rtol, atol=0)


def test_get_df_from_directory_dataset_test_fractions(dataset):
    grouped = (
        dataset("ids2018test")[["Flow Duration", "Label"]]
        .groupby("Label")
        .agg(count=("Flow Duration", "count"))
    )
    reference = (
        pd.DataFrame.from_dict(LABEL_FRACTIONS, orient="index")
        .reset_index()
        .rename(columns={"index": "Label", 0: "count"})
    ).set_index("Label")
    result = (
        (grouped["count"] / grouped["count"].sum()).reset_index().set_index("Label")
    )
    for label in sorted(LABEL_FRACTIONS.keys()):
        if label in ["Brute Force -XSS", "SQL Injection"]:
            rtol = 0.3
        else:
            rtol = 0.1
        assert np.isclose(result.loc[label], reference.loc[label], rtol=rtol, atol=0)


def test_get_df_from_directory_dataset_validate_fractions(dataset):
    grouped = (
        dataset("ids2018validate")[["Flow Duration", "Label"]]
        .groupby("Label")
        .agg(count=("Flow Duration", "count"))
    )
    reference = (
        pd.DataFrame.from_dict(LABEL_FRACTIONS, orient="index")
        .reset_index()
        .rename(columns={"index": "Label", 0: "count"})
    ).set_index("Label")
    result = (
        (grouped["count"] / grouped["count"].sum()).reset_index().set_index("Label")
    )
    for label in sorted(LABEL_FRACTIONS.keys()):
        if label in ["Brute Force -XSS", "SQL Injection"]:
            rtol = 0.3
        else:
            rtol = 0.1
        assert np.isclose(result.loc[label], reference.loc[label], rtol=rtol, atol=0)
