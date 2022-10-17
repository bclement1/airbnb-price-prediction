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

import sklearn as skl
from sklearn.model_selection import train_test_split

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
    DATA_FOLDER = "data"
    PATH_CSV = "train_airbnb_berlin.csv"
    # Load training data
    df_data = pd.read_csv(os.path.join(os.getcwd(), DATA_FOLDER, PATH_CSV))

    # Direct Processing
    df_data.replace("*", np.nan, inplace=True)
    for col in datetime_columns:
        df_data[col] = pd.to_datetime(df_data[col])
    df_data["Host Response Rate"] = df_data["Host Response Rate"].apply(treatment)
    df_data[numerical_features] = df_data[numerical_features].applymap(float)
    return df_data


def extract_y_data(df: pd.DataFrame):
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
            "Country",
            "Listing ID",
            "Listing Name",
            "Host ID",
            "Host Name",
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
        le = skl.preprocessing.LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
    return df


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def preprocessing(df: pd.DataFrame, encoder_method="LabelEncoder"):
    """
    Function applying all the previous preprocessing functions.
    """
    (data, df_identifiers) = remove_cols(df)
    X, y = extract_y_data(data)
    if encoder_method == "LabelEncoder":
        X = LabelEncoder_df(X, categorical_columns)
    (X_train, X_test, y_train, y_test) = split_data(X, y)
    return (X_train, y_train, X_test, y_test, df_identifiers)
