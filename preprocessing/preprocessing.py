"""
Preprocessing.py
"""

# Python ≥3.5 is required
import sys
from tkinter import Y

assert sys.version_info >= (3, 5)
import os

# Scikit-Learn ≥0.20 is required
import sklearn

assert sklearn.__version__ >= "0.20"

# Numpy arrays are used to store training and test data.
import numpy as np

# Pandas is used to manipulate tabular data.
import pandas as pd

# Ignore useless warnings (see SciPy issue #5998).
import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")

datetime_columns = ["Host Since", "First Review", "Last Review"]
categorical_columns = [
    "Is Superhost",
    "neighbourhood",
    "Neighborhood Group",
    "Is Exact Location",
    "Property Type",
    "Room Type",
    "Instant Bookable",
    "Business Travel Ready",
    "Host Response Time",
]
numerical_features = [
    "Postal Code",
    "Accomodates",
    "Bathrooms",
    "Beds",
    "Bedrooms",
    "Guests Included",
    "Min Nights",
]


def treatment(x):
    if type(x) == float:
        return x
    else:
        x = x.replace("%", "")
        return float(x) / 100


def load_data():
    """
    Desc
    """
    DATA_FOLDER = "data/"
    PATH_CSV = "train_airbnb_berlin.csv"

    # Load training data
    df_data = pd.read_csv(
        os.path.join(os.path.dirname(os.getcwd()), DATA_FOLDER, PATH_CSV)
    )

    # Direct Processing
    for col in datetime_columns:
        df_data[col] = pd.to_datetime(df_data[col])
    df_data["Host Response Rate"] = df_data["Host Response Rate"].apply(treatment)
    df_data[numerical_features] = df_data[numerical_features].applymap(float)
    print(df_data.head())
    print(df_data.shape)
    return (training_data,)


def split_data(df: pd.DataFrame):
    """
    Desc
    """
    y = df["Price"]
    X = df.drop(columns="Price")
    return (X, y)


def remove_cols(df: pd.DataFrame):
    """
    Remove columns that were proved not useful for the study.
    """
    df_identifiers = df[["Listing ID", "Listing Name", "Host ID"]]
    df.drop(
        columns=[
            "Listing Name",
            "City",
            "Country Code",
            "Code",
            "Listing ID",
            "Listing Name",
            "Host ID",
        ],
        inplace=True,
    )  # those columns only take a single value

    # those columns are useful for identifying the samples
    return (df, df_identifiers)


def categorical_features_handler():
    return 0


def numerical_features_scaler():
    return 0


def LabelEncoder_df(df, categorical_columns=categorical_columns):
    for col in categorical_columns:
        le = preprocessing.LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])


def preprocessing(df: pd.DataFrame):
    """
    Function applying all the previous preprocessing functions.
    """
    (X, y) = split_data(df)
    (X, df_identifiers) = remove_cols(X)
    return (X, y, df_identifiers)


if __name__ == "__main__":
    training_data, test_data = load_data()
    (X_train, y_train) = split_data(training_data)
    (X_test, y_test) = split_data(test_data)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
