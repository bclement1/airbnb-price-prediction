"""
Preprocessing.py
"""

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Ignore useless warnings (see SciPy issue #5998).
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Import some Transformers from sklearn, as well as function useful for handling data.
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# Numpy arrays are used to store training and test data.
import numpy as np

# Pandas is used to manipulate tabular data.
import pandas as pd

# import the function used to load the data
from load_data import load_data

# GLOBAL VARIABLES

DATETIME_COLUMNS = [
    "Host Since",
    "First Review",
    "Last Review"
]

CATEGORICAL_COLUMNS = [
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

NOMINAL_FEATURES = [
    "Is Superhost",
    "neighbourhood",
    "Neighborhood Group",
    "Is Exact Location",
    "Property Type",
    "Instant Bookable",
    "Business Travel Ready",
    
]

ORDINAL_FEATURES = [
    "Room Type", # indeed, one can consider that 'Private room' < 'Entire home'
    "Host Response Time"
]

NUMERICAL_FEATURES = [
    "Postal Code",
    "Accomodates",
    "Bathrooms",
    "Beds",
    "Bedrooms",
    "Guests Included",
    "Min Nights",
]

UNIVALUED_COLUMNS = [ # these columns take only a single, therefore are not useful for predicting the target
    "Country",
    "Country Code",
    "Country",
    "City"
]

IDENTIFIERS_COLUMNS = [ # these columns are only useful for identifying the samples
    "Listing ID",
    "Listing Name",
    "Host ID",
    "Host Name"
]

# CUSTOM ERRORS

class UnknownFillMethodError(Exception):
    pass

class UnknownEncodingMethodError(Exception):
    pass

class UnalignedDataFramesError(Exception):
    pass

# AUXILIARY PREPROCESSING FUNCTIONS

def insert_nan(df: pd.DataFrame):
    """
    Desc.
    """
    df.replace("*", np.nan, inplace=True)


def remove_samples_with_missing_target(df: pd.DataFrame):
    """
    Desc.
    """
    na_index = df[df["Price"].isna()].index
    df.drop(na_index, inplace=True, errors="ignore")


def process_datetime_columns(df: pd.DataFrame):
    """
    Desc.
    """
    for col in DATETIME_COLUMNS:
        df[col] = pd.to_datetime(df[col])
    return df


def datetime_to3columns(df: pd.DataFrame):
    for col in DATETIME_COLUMNS:
        df[col + "_year"] = df[col].apply(lambda x: x.year)
        df[col + "_month"] = df[col].apply(lambda x: x.month)
        df[col + "_day"] = df[col].apply(lambda x: x.day)
    df.drop(columns=DATETIME_COLUMNS, inplace=True)


def host_response_rate_transform(x):
    if type(x) == float:
        return x
    else:
        x = x.replace("%", "")
        return float(x) / 100


def handle_host_response_rate(df: pd.DataFrame):
    """
    This column is special. It is numerical but involved percentages. We need to format it apart.
    """
    df["Host Response Rate"] = df["Host Response Rate"].apply(host_response_rate_transform)
    return df


def ensure_numerical_columns_type(df):
    """
    Make sure all numerical columns are of type float.
    """
    df[NUMERICAL_FEATURES] = df[NUMERICAL_FEATURES].applymap(float)
    return df


def drop_univalued_columns(df: pd.DataFrame):
    """
    Desc.
    """
    df.drop(columns=UNIVALUED_COLUMNS, inplace=True)


def extract_identifiers(df: pd.DataFrame):
    """
    Remove columns that were proved not useful for the study.
    """
    df_identifiers = df[IDENTIFIERS_COLUMNS]
    df.drop(columns=IDENTIFIERS_COLUMNS, inplace=True)
    
    return (df, df_identifiers)


def extract_target(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Desc
    """
    y = df["Price"]
    X = df.drop(columns="Price")
    return (X, y)


def fill_na_mean(X: pd.DataFrame):
    """
    Desc.
    """
    for col in NUMERICAL_FEATURES:
        mean = X[col].mean()
        X[col].apply(lambda x: mean if(pd.isna(x)) else x)
    return X


def fill_na_median(X: pd.DataFrame):
    """
    Desc.
    """
    for col in NUMERICAL_FEATURES:
        med = X[col].median()
        X[col].apply(lambda x: med if(pd.isna(x)) else x)
    return X


def apply_one_hot_encoding(X: pd.DataFrame, full=False):
    """
    Apply one-hot encoding for categorical features that are nominal.
    """
    if(full):
        # select all categorical features
        X_to_encode = X[CATEGORICAL_COLUMNS]
        # apply pandas's get_dummies function on all categorical features
        X_encoded = pd.get_dummies(X_to_encode)
        # put everything back in a single DataFrame
        X = X_encoded
    else:
        # select the subset of nominal features
        X_to_encode = X[NOMINAL_FEATURES]
        X_numerical_and_ordinal = X.drop(columns=NOMINAL_FEATURES)
        # apply pandas's get_dummies function on the nominal features only
        X_encoded = pd.get_dummies(X_to_encode)
        # put everything back in a single DataFrame
        X = pd.concat([X_numerical_and_ordinal, X_encoded], axis=1)

    return X


def apply_label_encoding(X: pd.DataFrame, full=False):
    """
    Apply a Label encoding on the ordinal features. If full is set to True, apply the Label encoding onto all the
    categorical features.
    WARNING: applying a label encoding to all categorical features means in particular applying it to the nominal
    features, which is a mistake if you want to feed the data to - for example - a LinearRegression() model!
    See the lectures on this topic.
    Nonetheless, for particular models such as Regression Trees and Random Forests where the weak learner is a Regression
    Tree, one can use the Label encoding on all categorical features, since Trees can handle it.
    """
    
    if(full): 
        # apply the label encoding on all features
        cols_to_process = CATEGORICAL_COLUMNS
    else:
        # apply the label encoding on ordinal features only
        cols_to_process = ORDINAL_FEATURES
    for col in cols_to_process:
        if col == "Host Response Time": # specify the order to use for ordinal encoding, since it means something
           pass
        elif col == "Room Type":
            pass
        else:
            pass

        encoder = OrdinalEncoder()
        encoded_col = encoder.fit_transform(X[[col]])
        X[[col]] = encoded_col.reshape(-1, 1)

    return X
    

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Desc.
    """
    y = np.ravel(y)
    if X.shape[0] != y.shape[0]:
        raise UnalignedDataFramesError
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# MAIN PREPROCESSING FUNCTION

def preprocessing(df: pd.DataFrame, 
                  encoder_method="classical",
                  fill_na_method="mean"):
    """
    Function applying all the previously defined preprocessing functions.
    """
    # first, replace the '*' characters by proper NaN
    insert_nan(df)

    # then, remove samples with a missing y-value (missing target)
    remove_samples_with_missing_target(df)

    # then, convert columns involving dates to the proper format
    df = process_datetime_columns(df)

    # and split each date-based column into 3 components (day, month, year)
    datetime_to3columns(df)

    # apply a special treatment to the host_response_rate column, which involves percentages
    df = handle_host_response_rate(df)

    # ensure all numerical columns are of type float
    df = ensure_numerical_columns_type(df)

    # drop the columns that take only a single value
    drop_univalued_columns(df)

    # extract the identifiers from the data (columns needed to identify a given sample, like ID, Name...)
    (data, df_identifiers) = extract_identifiers(df)

    # extract the target variable from the data
    X, y = extract_target(data)

    # at this step, we fill the missing values in the features. 2 strategies are allowed
    if fill_na_method not in ["mean", "median"]:
        raise UnknownFillMethodError
    elif fill_na_method == "mean":
        X = fill_na_mean(X)
    elif fill_na_method == "median":
        X = fill_na_median(X)
    
    # now, we apply encoding to the categorical features
    if encoder_method not in ["classical", "FullLabelEncoder", "FullOneHotEncoder"]:
        raise UnknownEncodingMethodError
    elif encoder_method == "classical":
        X = apply_one_hot_encoding(X, full=False)
        X = apply_label_encoding(X, full=False)
    # WARNING: do not use the following two if you do not know what you are doing, use the default value instead
    elif encoder_method == "FullOneHotEncoder": # WARNING: be very careful when we use the following
        X = apply_one_hot_encoding(X, full=True)
    elif encoder_method == "FullLabelEncoder": # WARNING: be very careful when we use the following
        X = apply_label_encoding(X, full=True)
    
    # last, we split the features and target into a training and a test set
    (X_train, X_test, y_train, y_test) = split_data(X, y)

    return (X_train, y_train, X_test, y_test, df_identifiers)


if __name__ == "__main__":
    # load the data
    df = load_data()
    # apply the preprocessing
    (X_train, X_test, y_train, y_test, df_identifiers) = preprocessing(df)
    # ensure everything went fine
    print(X_train.head())

"""
TO DO:
Remove NaN from all columns using Imputers
Check is binary cols like 'Instant bookable' can be encoded using O:1 only instead of using 2 columns
Scale features!!!
Check again all cols before PCA
"""