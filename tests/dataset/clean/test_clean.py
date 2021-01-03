import pathlib
import inspect
import numpy as np
import pandas as pd
import pandas.testing
import pytest

from nd00333.dataset.clean import clean


def test_get_clean_df_remove_timestamp():
    data = [["2018"]]
    columns = ["Timestamp"]
    df_inp = pd.DataFrame(data=data, columns=columns)
    df_out = clean.get_clean_df(df_inp)
    assert list(df_out.columns) == []


def test_get_clean_df_remove_categorical_variables():
    data = [[0, 0, 0, 0, 0]]
    columns = ["Protocol", "Src IP", "Src Port", "Dst Port", "Dst IP"]
    df_inp = pd.DataFrame(data=data, columns=columns)
    df_out = clean.get_clean_df(df_inp)
    assert list(df_out.columns) == []


def test_get_clean_df_remove_custom_minus_1_values():
    data = [
        [0, 0, -1],
        [0, -1, 0],
        [-1, 0, 0],
        [-1, -1, 0],
    ]
    columns = ["Init Fwd Win Byts", "Init Bwd Win Byts", "dummy"]
    df_inp = pd.DataFrame(data=data, columns=columns)
    df_out = clean.get_clean_df(df_inp)
    pandas.testing.assert_frame_equal(
        df_out,
        pd.DataFrame(
            # only affect the custom columns
            data=[[0, 0, -1]],
            columns=columns,
        ),
    )


def test_get_clean_df_replace_custom_smaller_than_minus_1_values_with_0():
    data = [
        [-2, -1, -2],
        [-1, -1, -2],
        [-1, -2, -2],
        [-2, -2, -2],
    ]
    columns = ["Flow IAT Min", "Fwd IAT Min", "dummy"]
    df_inp = pd.DataFrame(data=data, columns=columns)
    df_out = clean.get_clean_df(df_inp)
    pandas.testing.assert_frame_equal(
        df_out,
        pd.DataFrame(
            # only affect the custom columns
            data=[[0, -1, -2], [-1, -1, -2], [-1, 0, -2], [0, 0, -2]],
            columns=columns,
        ),
    )


def test_get_clean_df_convert_all_columns_to_integers():
    data = [["1", "1.1"]]
    columns = ["dummy1", "dummy2"]
    df_inp = pd.DataFrame(data=data, columns=columns)
    df_out = clean.get_clean_df(df_inp)
    assert list(df_out.dtypes) == [np.dtype("int"), np.dtype("int")]


def test_get_clean_df_dropna_rows():
    data = [[0], [np.nan], [-np.inf], [np.inf]]
    columns = ["dummy"]
    df_inp = pd.DataFrame(data=data, columns=columns)
    df_out = clean.get_clean_df(df_inp)
    pandas.testing.assert_frame_equal(
        df_out,
        pd.DataFrame(
            data=[[0]],
            columns=columns,
        ),
    )
