"""
Scoring script
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame(
    {
        "Flow Duration": pd.Series([0], dtype="int64"),
        "TotLen Fwd Pkts": pd.Series([0], dtype="int64"),
        "TotLen Bwd Pkts": pd.Series([0], dtype="int64"),
        "Fwd Pkt Len Std": pd.Series([0], dtype="int64"),
        "Bwd Pkt Len Max": pd.Series([0], dtype="int64"),
        "Bwd Pkt Len Std": pd.Series([0], dtype="int64"),
        "Flow Byts/s": pd.Series([0], dtype="int64"),
        "Flow Pkts/s": pd.Series([0], dtype="int64"),
        "Flow IAT Max": pd.Series([0], dtype="int64"),
        "Bwd IAT Min": pd.Series([0], dtype="int64"),
        "Bwd Header Len": pd.Series([0], dtype="int64"),
        "Pkt Len Max": pd.Series([0], dtype="int64"),
        "Pkt Len Std": pd.Series([0], dtype="int64"),
        "RST Flag Cnt": pd.Series([0], dtype="int64"),
        "PSH Flag Cnt": pd.Series([0], dtype="int64"),
        "ECE Flag Cnt": pd.Series([0], dtype="int64"),
        "Init Fwd Win Byts": pd.Series([0], dtype="int64"),
        "Init Bwd Win Byts": pd.Series([0], dtype="int64"),
        "Fwd Seg Size Min": pd.Series([0], dtype="int64"),
    }
)
output_sample = np.array(["Benign"], dtype=object)


def init():
    """
    Initialize the model
    """
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)


@input_schema("data", PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    """
    Return prediction result for data
    """
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as exc:
        result = str(exc)
        return json.dumps({"error": result})
