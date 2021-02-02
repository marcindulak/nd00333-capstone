"""
Batch test the scoring endpoint
"""

import argparse
import glob
import json
import requests

import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import recall_score


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scoring-uri",
        required=True,
        type=str,
        help="The URI of the scoring service",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        type=str,
        help="The API key of the scoring service",
    )
    parser.add_argument(
        "--dataset-path",
        required=False,
        type=str,
        default="datasets/ids2018test",
        help="The path to the test dataset directory",
    )
    parser.add_argument(
        "--batch-size",
        required=False,
        type=int,
        default=100,
        help="The batch size used in post requests to the scoring service",
    )
    parser.add_argument(
        "--sample-size",
        required=False,
        type=int,
        default=None,
        help="The sample size of the data set used in the test",
    )

    return parser.parse_args()


def main(args):
    """
    Test
    """

    if not args.sample_size:
        sample_size = -1
    else:
        sample_size = args.sample_size

    csv_files = sorted(list(glob.glob(f"{args.dataset_path}/*csv")))
    df = pd.concat((pd.read_csv(csv) for csv in csv_files)).reset_index(drop=True)[:sample_size]
    x_test, y_test = df.drop(labels=["Label"], axis=1), df["Label"]

    headers = {"Content-Type": "application/json"}
    headers["Authorization"] = f"Bearer {args.api_key}"

    y_test_predict = []
    batch_size = args.batch_size
    batch_number_total = x_test.shape[0] // batch_size
    for batch_number, batch_start in enumerate(range(0, x_test.shape[0], batch_size)):
        if batch_number % 100 == 0:
            print(f"batch number {batch_number} of {batch_number_total}")
        data = x_test.iloc[batch_start : batch_start + batch_size, :].to_dict(
            orient="records"
        )
        x_test_sample = json.dumps({"data": data})
        response = requests.post(args.scoring_uri, x_test_sample, headers=headers)
        result = json.loads(response.json())["result"]
        y_test_predict.extend(result)

    metrics = recall_score(y_true=y_test, y_pred=y_test_predict, average="macro")
    print(f"recall_score {metrics}")

    report = classification_report(
        digits=4, y_true=y_test, y_pred=y_test_predict, output_dict=False
    )
    print(f"classification_report {report}")

    assert metrics > 0.84, f"Error: metrics {metrics} <= 0.84"

    return metrics, report


if __name__ == "__main__":
    _, _ = main(parse_args())
