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

def load_data():
    """
    Desc
    """
    DATA_FOLDER = "data/"
    TRAINING_CSV = "train_airbnb_berlin.csv"
    TEST_CSV = "test_airbnb_berlin.csv"

    # Load training data
    training_data = pd.read_csv(os.path.join(
                    os.path.dirname(os.getcwd()),
                    DATA_FOLDER,
                    TRAINING_CSV
                    )
                )     
    test_data = pd.read_csv(os.path.join(
                    os.path.dirname(os.getcwd()),
                    DATA_FOLDER,
                    TEST_CSV
                    )
                )
    print(training_data.head())
    print(training_data.shape, test_data.shape)
    return training_data, test_data

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
    df_identifiers = df[['Listing ID', 'Listing Name', 'Host ID']]
    df.drop(columns=[
        'Listing Name', 
        'City',
        'Country Code',
        'Code',
        'Listing ID',
        'Listing Name',
        'Host ID'
        ],
        inplace=True
    ) # those columns only take a single value
    
    # those columns are useful for identifying the samples
    return (df, df_identifiers)

def categorical_features_handler():
    return 0

def numerical_features_scaler():
    return 0

def preprocessing(df: pd.DataFrame):
    """
    Function applying all the previous preprocessing functions.
    """
    (X, y) = split_data(df)
    (X, df_identifiers) = remove_cols(X)
    return (X, y, df_identifiers)


if __name__ == "__main__":
    training_data, test_data = load_data()
    (X_train, X_test, y_train, y_test) = split_data(training_data, test_data)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)