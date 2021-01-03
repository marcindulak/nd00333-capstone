import argparse
import pathlib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from nd00333.dataset.load import load
from nd00333 import utils as package_utils

logger = package_utils.get_logger()

DEFAULT_ARGS = {
    "--dataset-path": "datasets",
    "--dataset-name": "ids2018full",
    "--dataset-name-train": "train",
    "--dataset-name-test": "test",
    "--random-seed": 0,
    "--sample-fraction": False,
    "--save-to-disk": False,
    "--target-label": "target",
    "--test-size": 0.5,
}


def parse_args():
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
        "--dataset-name-train",
        required=False,
        type=str,
        default=DEFAULT_ARGS["--dataset-name-train"],
        help="The nname used for naming the train dataset",
    )
    parser.add_argument(
        "--dataset-name-test",
        required=False,
        type=str,
        default=DEFAULT_ARGS["--dataset-name-test"],
        help="The nname used for naming the test dataset",
    )
    parser.add_argument(
        "--random-seed",
        required=False,
        type=int,
        default=DEFAULT_ARGS["--random-seed"],
        help="Random seed",
    )
    parser.add_argument(
        "--sample-fraction",
        required=False,
        type=float,
        default=DEFAULT_ARGS["--sample-fraction"],
        help="The fraction of sampled data to use",
    )
    parser.add_argument(
        "--save-to-disk",
        required=False,
        action="store_true",
        default=DEFAULT_ARGS["--save-to-disk"],
        help="Whether to save the resulting datasets to disk",
    )
    parser.add_argument(
        "--target-label",
        required=False,
        type=str,
        default=DEFAULT_ARGS["--target-label"],
        help="The target label",
    )
    parser.add_argument(
        "--test-size",
        required=False,
        type=float,
        default=DEFAULT_ARGS["--test-size"],
        help="The fraction of data points used for the test set",
    )
    return parser.parse_args()


def split(args):

    directory = pathlib.Path(args.dataset_path, args.dataset_name)
    if not directory.exists():
        msg = f"The dataset directory {directory} does not exist"
        logger.exception(msg)
        raise RuntimeError(msg)

    df = load.get_df_from_directory(directory)

    if args.sample_fraction:
        df = df.sample(
            frac=args.sample_fraction, replace=False, random_state=args.random_seed
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(labels=[args.target_label], axis=1),
        df[args.target_label],
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    def save_to_disk(X, y, name):
        if name == "train":
            directory = pathlib.Path(args.dataset_path, f"{args.dataset_name_train}")
        else:
            directory = pathlib.Path(args.dataset_path, f"{args.dataset_name_test}")
        directory.mkdir(parents=False, exist_ok=False)
        pd.concat([X, y], axis=1).to_csv(
            pathlib.Path(directory, "data.csv"), index=False
        )

    if args.save_to_disk:
        save_to_disk(X_train, y_train, "train")
        save_to_disk(X_test, y_test, "test")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    args = parse_args()
    _, _, _, _ = split(args)
