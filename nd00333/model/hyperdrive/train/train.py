"""
Training script
"""

import argparse
import glob
import logging
import pathlib

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score

from azureml.core import Run

from pythonjsonlogger import jsonlogger


def get_logger():
    """
    Log json. It's a duplicate function defintion to keep train.py self-contained
    """
    logger = logging.getLogger()

    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


logger = get_logger()

run = Run.get_context()


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-train-path",
        required=False,
        type=str,
        default="train",
        help="The path to the training data directory",
    )
    parser.add_argument(
        "--dataset-validate-path",
        required=False,
        type=str,
        default="validate",
        help="The path to the validation data directory",
    )
    parser.add_argument(
        "--dataset-is-remote",
        required=False,
        action="store_true",
        default=False,
        help="Load the dataset from AzureML datastore instead of the local disk",
    )
    parser.add_argument(
        "--model-filename",
        required=False,
        type=str,
        default="model.pkl",
        help="Filename of the model pickle",
    )
    parser.add_argument(
        "--hyperparameter-n_estimators",
        required=False,
        type=int,
        default=100,
        help="",
    )
    parser.add_argument(
        "--hyperparameter-criterion",
        required=False,
        type=str,
        default="gini",
        help="",
    )
    parser.add_argument(
        "--hyperparameter-max_depth",
        required=False,
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "--hyperparameter-min_samples_split",
        required=False,
        type=int,
        default=2,
        help="",
    )
    parser.add_argument(
        "--hyperparameter-min_samples_leaf",
        required=False,
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--hyperparameter-min_weight_fraction_leaf",
        required=False,
        type=float,
        default=0.0,
        help="",
    )
    parser.add_argument(
        "--hyperparameter-max_features",
        required=False,
        type=str,
        default="auto",
        help="",
    )
    parser.add_argument(
        "--hyperparameter-max_leaf_nodes",
        required=False,
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "--hyperparameter-min_impurity_decrease",
        required=False,
        type=float,
        default=0.0,
        help="",
    )
    parser.add_argument(
        "--hyperparameter-bootstrap",
        required=False,
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--hyperparameter-oob_score",
        required=False,
        type=bool,
        default=False,
        help="",
    )
    parser.add_argument(
        "--hyperparameter-n_jobs",
        required=False,
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "--hyperparameter-random_state",
        required=False,
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "--hyperparameter-ccp_alpha",
        required=False,
        type=float,
        default=0.0,
        help="",
    )
    parser.add_argument(
        "--hyperparameter-max_samples",
        required=False,
        type=int,
        default=None,
        help="",
    )

    return parser.parse_args()


def get_x_y(dataset_path):
    """
    Return a dataset split into feature and target
    """
    if pathlib.Path(dataset_path).is_dir():
        csv_files = sorted(list(glob.glob(f"{dataset_path}/*csv")))
    else:
        csv_files = [dataset_path]
    dataframe = pd.concat((pd.read_csv(csv) for csv in csv_files)).reset_index(
        drop=True
    )
    features, target = dataframe.drop(labels=["Label"], axis=1), dataframe["Label"]
    return features, target


def main(args):
    """
    Main
    """
    logger.info(msg="main", extra={"arguments": vars(args)})

    # The outputs directory is automatically stored in the AzureML datastore
    outputs = pathlib.Path("outputs")
    outputs.mkdir(parents=False, exist_ok=True)
    model_filename = pathlib.Path(outputs, args.model_filename)
    if model_filename.exists():
        msg = f"The model file {model_filename} already exists"
        logger.exception(msg)
        raise RuntimeError(msg)

    # Build hyperparameters dictionary and log the values
    hyperparameters = {}
    for arg, value in (vars(args)).items():
        if arg.startswith("hyperparameter"):
            hyperparameter = arg.replace("hyperparameter_", "")
            hyperparameters[hyperparameter] = value
            run.log(hyperparameter, value)

    logger.info(msg="main", extra={"hyperparameters": hyperparameters})
    model = RandomForestClassifier(**hyperparameters)

    x_train, y_train = get_x_y(args.dataset_train_path)
    x_validate, y_validate = get_x_y(args.dataset_validate_path)

    model.fit(x_train, y_train)

    y_validate_predict = model.predict(x_validate)
    metrics = recall_score(
        y_true=y_validate, y_pred=y_validate_predict, average="macro"
    )
    logger.info(
        msg="main",
        extra={
            "norm_macro_recall": metrics,
        },
    )
    run.log("norm_macro_recall", metrics)

    report = classification_report(
        digits=4, y_true=y_validate, y_pred=y_validate_predict, output_dict=True
    )
    logger.info(
        msg="main",
        extra={
            "classification_report": report,
        },
    )

    joblib.dump(model, model_filename)
    logger.info(
        msg="main",
        extra={
            "model_filename": model_filename,
        },
    )
    run.log("model_filename", model_filename)

    return model, metrics, report


if __name__ == "__main__":
    main(parse_args())
