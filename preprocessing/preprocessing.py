"""
Preprocessing.py
"""

# Python ≥3.5 is required
from audioop import minmax
import sys
assert sys.version_info >= (3, 5)
import os

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Numpy arrays are used to store training and test data.
import numpy as np

# Pandas is used to manipulate tabular data.
import pandas as pd

# Ignore useless warnings (see SciPy issue #5998).
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# GLOBAL VARIABLES

# List of the categorical features
categorical_features = [
    "Is Superhost",
    "neighbourhood",
    "Neighborhood Group",
    "Is Exact Location",
    "Property Type",
    "Room Type",
    "Instant Bookable",
    "Business Travel Ready",
    "Host Response Time"
]

# Functions
def load_data():
    """
    Desc
    """
    DATA_FOLDER = "data/"
    TRAINING_CSV = "train_airbnb_berlin.csv"
    TEST_CSV = "test_airbnb_berlin.csv"

    # Load training data
    df = pd.read_csv(os.path.join(
                    os.path.dirname(os.getcwd()),
                    DATA_FOLDER,
                    TRAINING_CSV
                    )
                )
    """
    test_data = pd.read_csv(os.path.join(
                    os.path.dirname(os.getcwd()),
                    DATA_FOLDER,
                    TEST_CSV
                    )
                )
    """    
    
    # print(df.head())
    # print(df.shape)
    return df

def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Desc
    """
    y = df["Price"]
    X = df.drop(columns="Price")
    X_train, y_train, X_test, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state
    )
    
    return (X_train, y_train, X_test, y_test)

def remove_cols(X: pd.DataFrame):
    """
    Remove columns that were proved not useful for the study.
    """
    df_identifiers = X[['Listing ID', 'Listing Name', 'Host ID', 'Host Name']]
    X.drop(columns=[
        'City',
        'Country Code',
        'Country',
        'Listing ID',
        'Listing Name',
        'Host ID',
        'Host Name'
        ],
        inplace=True
    ) # those columns only take a single value
    
    # those columns are useful for identifying the samples
    return (X, df_identifiers)

def one_hot_encoding(df: pd.DataFrame, categorical_features: list):
    """
    Apply one-hot encoding 
    """
    # We get the list of the numerical features.
    df_categorical = df[categorical_features]
    df.drop(columns=categorical_features, inplace=True)
    print(df_categorical.head())
    # Apply pandas's get_dummies function on the categorical features
    df_categorical_encoded = pd.get_dummies(df_categorical)
    print(df_categorical_encoded.head())
    
    df = pd.concat([df, df_categorical_encoded], axis=1)
    return df

def numerical_features_scaler():
    return 0

def preprocessing(df: pd.DataFrame):
    """
    Function applying all the previous preprocessing functions.
    """
    (df_clean, df_identifiers) = remove_cols(df)
    df_encoded = one_hot_encoding(df_clean, categorical_features=categorical_features)

    return (df_encoded, df_identifiers)


if __name__ == "__main__":
    df = load_data() # full data
    # print(df.shape)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    df_clean, df_identifiers = preprocessing(df)

    (X_train, X_test, y_train, y_test) = split_data(df_clean)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
