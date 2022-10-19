"""
Preprocessing.py
"""

# Python ≥3.5 is required
from audioop import minmax
import sys
from tkinter import Y

assert sys.version_info >= (3, 5)
import os

# Scikit-Learn ≥0.20 is required
import sklearn

assert sklearn.__version__ >= "0.20"

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#Scaler from sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


# Numpy arrays are used to store training and test data.
import numpy as np

# Pandas is used to manipulate tabular data.
import pandas as pd

# Ignore useless warnings (see SciPy issue #5998).
import warnings

import sklearn as skl
from sklearn.model_selection import train_test_split

import git

warnings.filterwarnings(action="ignore", message="^internal gelsd")

# Global variables
datetime_columns = [
    "Host Since",
    "First Review",
    "Last Review"
]
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


def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


def load_data():
    """
    Desc
    """
    DATA_FOLDER = "data"
    PATH_CSV = "train_airbnb_berlin.csv"
    # Load training data
    df_data = pd.read_csv(
        os.path.join(get_git_root(os.getcwd()), DATA_FOLDER, PATH_CSV)
    )

    # Direct Processing
    df_data.replace("*", np.nan, inplace=True)
    for col in datetime_columns:
        df_data[col] = pd.to_datetime(df_data[col])
    df_data["Host Response Rate"] = df_data["Host Response Rate"].apply(treatment)
    df_data[numerical_features] = df_data[numerical_features].applymap(float)
    
    na_index = df_data[df_data["Price"].isna()].index
    df_data.drop(na_index, inplace=True, errors="ignore")
    return df_data


def extract_y_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Desc
    """
    y = df["Price"]
    X = df.drop(columns="Price")
    return (X, y)

def remove_cols(X: pd.DataFrame):
    """
    Remove columns that were proved not useful for the study.
    """
    df_identifiers = X[['Listing ID', 'Listing Name', 'Host ID', 'Host Name']]
    # those columns are useful for identifying the samples

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
    
    return (X, df_identifiers)

def one_hot_encoding(df: pd.DataFrame, categorical_features: list):
    """
    Apply one-hot encoding for categorical nominal features.
    """
    # We get the list of the numerical features.
    df_categorical = df[categorical_features]
    df.drop(columns=categorical_features, inplace=True)
    # Apply pandas's get_dummies function on the categorical features
    df_categorical_encoded = pd.get_dummies(df_categorical)
    # Put everything back in a single DataFrame
    df = pd.concat([df, df_categorical_encoded], axis=1)
    return df


def numerical_features_scaler(X_train: pd.DataFrame, scale_method: str):
    """
    Apply a scale on the training features depends on the scale method choose.
    """

    method_to_scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "MaxAbsScaler": MaxAbsScaler(), "RobustScaler": RobustScaler(), "Normalizer": Normalizer(), "QuantileTransformer": QuantileTransformer(), "PowerTransformer": PowerTransformer()}

    if scale_method not in method_to_scaler.keys():
        raise Exception("This scaling method is not available. Please choose among [StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer]")

    scaler = method_to_scaler[scale_method]

    X_numerical = X_train.filter(numerical_features, axis=1)
    X_numerical = scaler.fit_transform(X_numerical)

    X_train = pd.concat([X_numerical, X_train[[datetime_columns]+[categorical_columns]]], axis=1)
    
    return X_train

def target_feature_scaler(y_train: pd.DataFrame, scale_method: str):
    """
    Apply a scale on the tarining target data depends on the scale method choose.
    """

    method_to_scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "MaxAbsScaler": MaxAbsScaler(), "RobustScaler": RobustScaler(), "Normalizer": Normalizer(), "QuantileTransformer": QuantileTransformer(), "PowerTransformer": PowerTransformer()}

    if scale_method not in method_to_scaler.keys():
        raise Exception("This scaling method is not available. Please choose among [StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer]")

    scaler = method_to_scaler[scale_method]
    y_train = scaler.fit_transform(y_train)

    return y_train



def fill_na_mean(df):
    for col in df:
        mean=df[col].mean()
        df[col][df[col].isna()]=mean
    return df

def fill_na_median(df):
    for col in df:
        med=df[col].median()
        df[col][df[col].isna()]=med
    return df

def LabelEncoder_df(df, categorical_columns=categorical_columns):
    for col in categorical_columns:
        le = skl.preprocessing.LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
    return df

def datetime_to3columns(df,datetime_columns=datetime_columns):
    for col in datetime_columns:
        df[col + "_year"] = df[col].apply(lambda x: x.year)
        df[col + "_month"] = df[col].apply(lambda x: x.month)
        df[col + "_day"] = df[col].apply(lambda x: x.day)
    df.drop(columns=datetime_columns, inplace=True)
    
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def preprocessing(df: pd.DataFrame, 
                  encoder_method=None,
                  fill_na_method=None,
                  datetime_treatment=None):
    """
    Function applying all the previous preprocessing functions.
    """
    (data, df_identifiers) = remove_cols(df)
    X, y = extract_y_data(data)
    if encoder_method == "LabelEncoder":
        X = LabelEncoder_df(X, categorical_columns)
        
    if fill_na_method == "Mean":
        X = fill_na_mean(X)
    elif fill_na_method == "Median":
        X = fill_na_median(X)
        
    if datetime_treatment=='Linearization':
        datetime_to3columns(X)
    
    (X_train, X_test, y_train, y_test) = split_data(X, y)
    X_train = numerical_features_scaler(X_train, 'method')
    y_train = target_feature_scaler(y_train, 'method')

    return (X_train, y_train, X_test, y_test, df_identifiers)

