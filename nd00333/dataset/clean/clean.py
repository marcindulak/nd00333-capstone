# -*- coding: utf-8 -*-
"""
Data preprocessing
"""

import numpy as np
import pandas as pd

from IPython.display import display
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report


def get_clean_df(df, display_max_rows=100, verbose=0):
    """
    Read from df and return a cleaned DataFrame
    """

    # Make a dataFrame copy
    df = df.copy(deep=True)

    if verbose > 0:
        print("DataFrame before cleaning")
        with pd.option_context(
            "display.max_rows", display_max_rows, "display.max_columns", None
        ):
            if verbose > 1:
                display(df.head().transpose())
            display(df.describe().transpose())

    columns = df.columns.tolist()
    # Remove Timestamp, since the events are considered independent
    for column in ["Timestamp"]:
        if column in columns:
            df.drop([column], axis=1, inplace=True)

    # Remove categorical variables
    for column in ["Protocol", "Src IP", "Src Port", "Dst Port", "Dst IP"]:
        if column in columns:
            df.drop([column], axis=1, inplace=True)

    # Replace values of -1 with nan
    for column in ["Init Fwd Win Byts", "Init Bwd Win Byts"]:
        if column in columns:
            df[column] = df[column].replace(-1, np.nan, inplace=False)

    # Replace negative values smaller than -1 with 0
    for column in ["Flow IAT Min", "Fwd IAT Min"]:
        if column in columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df[column].values[df[column].values < -1] = 0

    if verbose > 1:
        print(
            (
                "DataFrame after feature removal, conversion to np.nan and"
                + " replacing with negative values with 0"
            )
        )
        with pd.option_context(
            "display.max_rows", display_max_rows, "display.max_columns", None
        ):
            display(df.head().transpose())
            display(df.describe().transpose())

    # Convert all columns to_numeric, replacing errors with nan
    for column in sorted(df.columns.tolist()):
        if column not in ["Label"]:
            if verbose > 1:
                print(f"Converting column {column} to_numeric")
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if verbose > 1:
        print("DataFrame after to_numeric")
        with pd.option_context(
            "display.max_rows", display_max_rows, "display.max_columns", None
        ):
            display(df.head().transpose())
            display(df.describe().transpose())

    if verbose > 1:
        print("DataFrame missing values before their removal")
        with pd.option_context(
            "display.max_rows", display_max_rows, "display.max_columns", None
        ):
            display(df[df.isnull().any(axis=1)].describe().transpose())

    # Print the original DataFrame distribution of Labels
    if verbose > 0:
        df_shape_original = df.shape
        label = f"DataFrame before missing values removal, shape {df_shape_original}"
        print(label)
        df.groupby(["Label"]).size().plot(kind="bar", label=label)
        plt.show()
        display(df.groupby(["Label"]).size().reset_index(name="count"))
        display(
            df.groupby(["Label"])
            .size()
            .reset_index(name="count")
            .apply(lambda x: x["count"] / df_shape_original[0], axis=1)
        )
        plt.show()

    if verbose > 0:
        df_shape_only_missing_values = df[df.isnull().any(axis=1)].shape
        if df_shape_only_missing_values[0] > 0:
            label = (
                f"DataFrame only missing values, shape {df_shape_only_missing_values}"
            )
            print(label)
            df[df.isnull().any(axis=1)].groupby(["Label"]).size().plot(
                kind="bar", label=label
            )
            plt.show()
            display(
                df[df.isnull().any(axis=1)]
                .groupby(["Label"])
                .size()
                .reset_index(name="count")
            )
            display(
                df[df.isnull().any(axis=1)]
                .groupby(["Label"])
                .size()
                .reset_index(name="count")
                .apply(lambda x: x["count"] / df_shape_only_missing_values[0], axis=1)
            )
            plt.show()
        else:
            print(
                f"DataFrame has no missing values, shape {df_shape_only_missing_values}"
            )

    # Remove rows with missing values
    # Inifnite flow does not make sense, remove those rows
    with pd.option_context("mode.use_inf_as_na", True):
        df = df.dropna()

    # Convert all numeric features to integers
    for column in sorted(df.columns.tolist()):
        if column not in ["Label"]:
            if verbose > 1:
                print(f"Converting column {column} round(0).astype(int)")
            df[column] = df[column].round(0).astype(int)

    df_shape_after_missing_values_removal = df.shape
    if verbose > 0:
        label = (
            "DataFrame after missing values removal,"
            + f"shape {df_shape_after_missing_values_removal}"
        )
        print(label)
        df.groupby(["Label"]).size().plot(kind="bar", label=label)
        plt.show()
        display(df.groupby(["Label"]).size().reset_index(name="count"))
        display(
            df.groupby(["Label"])
            .size()
            .reset_index(name="count")
            .apply(
                lambda x: x["count"] / df_shape_after_missing_values_removal[0], axis=1
            )
        )
        plt.show()

    if verbose > 0:
        print(
            f"DataFrame after missing values removal, shape {df_shape_after_missing_values_removal}"
        )
        with pd.option_context(
            "display.max_rows", display_max_rows, "display.max_columns", None
        ):
            if verbose > 1:
                display(df.head().transpose())
            display(df.describe().transpose())

    return df


def get_feature_list(data, tolerance=0.0001, sample_fraction=None):
    """
    Return list of features which, when added one-by-one improve the metrics by tolerance.
    github.com/solegalli/feature-selection-for-machine-learning/tree/master/11-Hybrid-methods

    BSD 3-Clause License

    Copyright (c) 2018-2020, Soledad Galli
    Feature Selection for Machine Learning - Online Course:
    https://www.udemy.com/feature-selection-for-machine-learning


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
        list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
        this list of conditions and the following disclaimer in the documentation
        and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.


    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    # Define the model
    model_reference = RandomForestClassifier(
        n_estimators=20, max_depth=8, n_jobs=-1, random_state=0
    )

    # Convert target feature into integer
    targets = {value: key for key, value in enumerate(data["target"].unique())}

    print(f"targets {targets}")

    print("Initial DataFrame target")
    data["target"].describe()

    data["target"] = data.pop("target").apply(lambda s: targets[s])

    print("Final DataFrame target")
    data["target"].describe()

    if sample_fraction:
        data_sample = data.sample(frac=sample_fraction, replace=False, random_state=0)
    else:
        data_sample = data

    x_train, x_test, y_train, y_test = train_test_split(
        data_sample.drop(labels=["target"], axis=1),
        data_sample["target"],
        test_size=0.3,
        random_state=0,
    )

    print(f"Initial x_train.shape, x_test.shape {x_train.shape}, {x_test.shape}")

    quasi_constant_feat = []

    # iterate over every feature
    for feature in x_train.columns:

        # find the predominant value, that is the value that is shared
        # by most observations
        predominant = (
            (x_train[feature].value_counts() / np.float(len(x_train)))
            .sort_values(ascending=False)
            .values[0]
        )

        # evaluate the predominant feature: do more than 99.9% of the observations
        # show 1 value?
        if predominant > 0.999:

            # if yes, add the variable to the list
            quasi_constant_feat.append(feature)

    print(f"quasi_constant_feat {quasi_constant_feat}")

    x_train.drop(labels=quasi_constant_feat, axis=1, inplace=True)
    x_test.drop(labels=quasi_constant_feat, axis=1, inplace=True)

    print(
        (
            "After quasi_constant_feat removal x_train.shape, x_test.shape"
            + f" {x_train.shape}, {x_test.shape}"
        )
    )

    duplicated_feat = []
    for i in range(0, len(x_train.columns)):
        if i % 10 == 0:  # this helps me understand how the loop is going
            print(i)

        col_1 = x_train.columns[i]

        for col_2 in x_train.columns[i + 1 :]:
            if x_train[col_1].equals(x_train[col_2]):
                duplicated_feat.append(col_2)

    print(f"duplicated_feat {duplicated_feat}")

    x_train.drop(labels=duplicated_feat, axis=1, inplace=True)
    x_test.drop(labels=duplicated_feat, axis=1, inplace=True)

    print(
        f"After duplicated_feat removal x_train.shape, x_test.shape {x_train.shape}, {x_test.shape}"
    )

    model_full = model_reference

    model_full.fit(x_train, y_train)

    # calculate the performance metrics in the test set
    y_pred_test = model_full.predict(x_test)
    performance_metrics_full = recall_score(
        y_true=y_test, y_pred=y_pred_test, average="macro"
    )

    print("Test performance metrics=%f" % (performance_metrics_full))

    classification_report_full = classification_report(
        digits=4, y_true=y_test, y_pred=y_pred_test, output_dict=False
    )
    print("Test performance clasification report\n", classification_report_full)

    features = pd.Series(model_full.feature_importances_)
    features.index = x_train.columns

    # sort the features by importance
    features.sort_values(ascending=False, inplace=True)

    # plot
    features.plot.bar(figsize=(20, 6))
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.show()

    # make list of ordered features
    features = list(features.index)
    print(f"features importance list {features}")

    # next, we need to build a machine learning
    # algorithm using only the most important feature

    # build initial model with 1 feature
    model_one_feature = model_reference

    # train using only the most important feature
    model_one_feature.fit(x_train[features[0]].to_frame(), y_train)

    # calculate the performance metrics in the test set
    y_pred_test = model_one_feature.predict(x_test[features[0]].to_frame())

    performance_metrics_first = recall_score(
        y_true=y_test, y_pred=y_pred_test, average="macro"
    )

    print("Test one feature performance metrics=%f" % (performance_metrics_first))

    classification_report_first = classification_report(
        digits=4, y_true=y_test, y_pred=y_pred_test, output_dict=False
    )
    print(
        "Test one feature performance clasification report\n",
        classification_report_first,
    )

    print("doing recursive feature addition")

    # we initialise a list where we will collect the
    # features we should keep
    features_to_keep = [features[0]]

    # set a counter to know which feature is being evaluated
    count = 1

    # now we loop over all the features, in order of importance:
    # remember that features in the list are ordered
    # by importance
    for feature in features[1:]:
        print()
        print("testing feature: ", feature, count, " out of ", len(features))
        count = count + 1

        # initialise model
        model_int = model_reference

        # fit model with the selected features
        # and the feature to be evaluated
        model_int.fit(x_train[features_to_keep + [feature]], y_train)

        # make a prediction over the test set
        y_pred_test = model_int.predict(x_test[features_to_keep + [feature]])

        # calculate the new performance metrics
        performance_metrics_int = recall_score(
            y_true=y_test, y_pred=y_pred_test, average="macro"
        )
        print("New Test performance metrics={}".format((performance_metrics_int)))

        # print the original performance metrics with one feature
        print(
            "Previous round performance metrics={}".format((performance_metrics_first))
        )

        # determine the increase in the performance metrics
        diff_performance_metrics = performance_metrics_int - performance_metrics_first

        # compare the increase in performance metrics with the tolerance
        # we set previously
        if diff_performance_metrics >= tolerance:
            print("Increase in performance_metrics={}".format(diff_performance_metrics))
            print("keep: ", feature)
            print
            # if the increase in the performance metrics is bigger than the threshold
            # we keep the feature and re-adjust the performance metrics to the new value
            # considering the added feature
            performance_metrics_first = performance_metrics_int

            # and we append the feature to keep to the list
            features_to_keep.append(feature)
        else:
            # we ignore the feature
            print("Increase in performance metrics={}".format(diff_performance_metrics))
            print("skip: ", feature)
            print

    # now the loop is finished, we evaluated all the features
    print("DONE!!")
    print("total features to keep: ", len(features_to_keep))

    print(f"features_to_keep {features_to_keep}")

    # finally, let's compare performance of a model built using the selected
    # features vs the full model

    # build initial model
    model_final = model_reference

    # fit the model with the selected features
    model_final.fit(x_train[features_to_keep], y_train)

    # make predictions
    y_pred_test = model_final.predict(x_test[features_to_keep])

    # calculate performance metrics
    performance_metrics_final = recall_score(
        y_true=y_test, y_pred=y_pred_test, average="macro"
    )
    print("Test selected features performance metrics=%f" % (performance_metrics_final))

    classification_report_final = classification_report(
        digits=4, y_true=y_test, y_pred=y_pred_test, output_dict=False
    )
    print(
        "Test selected features performance clasification report\n",
        classification_report_final,
    )

    return features_to_keep
