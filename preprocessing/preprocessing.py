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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler


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
    "Host Response Time"
]

NOMINAL_FEATURES = [
    "neighbourhood",
    "Neighborhood Group",
    "Property Type"
]

BOOLEAN_FEATURES = [ # apply only a binary map 0:1 for these (take only 2 values)
    "Is Superhost",
    "Is Exact Location",
    "Instant Bookable",
    "Business Travel Ready"
]

ORDINAL_FEATURES = [
    "Room Type", # indeed, one can consider that 'Private room' < 'Entire home'
    "Host Response Time"
]

NUMERICAL_FEATURES = [
    "Accomodates",
    "Bathrooms",
    "Beds",
    "Bedrooms",
    "Guests Included",
    "Min Nights",
    "Reviews",
    "Overall Rating",
    "Accuracy Rating",
    "Cleanliness Rating",
    "Checkin Rating",
    "Communication Rating",
    "Location Rating",
    "Value Rating",
    "Host Response Rate",
    "Latitude",
    "Longitude",
    "Postal Code"
]

PERCENTAGE_COLUMNS = [
    "Host Response Rate"
]

UNIVALUED_COLUMNS = [ # these columns take only a single, therefore are not useful for predicting the target
    "Country", # only 'Germany'
    "Country Code", # only 'DE
    "City" # only 'nan' or '*' (replaced by np.nan) or 'Berlin'
]

IDENTIFIERS_COLUMNS = [ # these columns are only useful for identifying the samples
    "Listing ID",
    "Listing Name",
    "Host ID",
    "Host Name"
]

# CUSTOM ERRORS

class UnknownFillMethodError(Exception):
    """
    Desc.
    """
    pass

class UnknownEncodingMethodError(Exception):
    """
    Desc.
    """
    pass

class UnalignedDataFramesError(Exception):
    """
    Desc.
    """
    pass

class UnspecifiedMappingError(Exception):
    """
    Desc.
    """
    pass

class UnknownScalingMethodError(Exception):
    """
    Desc.
    """
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
    """
    Desc. + handle NaN value in each new column
    """
    for col in DATETIME_COLUMNS:
        df[col + "_year"] = df[col].apply(lambda x: x.year)
        # handle missing values in year column
        df[col + "_year"] = df[col + "_year"].apply(
            lambda x: int(df[col + "_year"].mean()) if pd.isna(x) else int(x)
        )
        # last, scale the new column so that it is ready for models
        df[col + "_year"] = (
            df[col + "_year"] - df[col + "_year"].min()
        ) / (
            df[col + "_year"].max() - df[col + "_year"].min()
        )

        df[col + "_month"] = df[col].apply(lambda x: x.month)
        # handle missing values in year column
        df[col + "_month"] = df[col + "_month"].apply(
            lambda x: int(df[col + "_month"].mean()) if pd.isna(x) else int(x)
        )
        # last, scale the new column so that it is ready for models
        df[col + "_month"] = (
            df[col + "_month"] - df[col + "_month"].min()
        ) / (
            df[col + "_month"].max() - df[col + "_month"].min()
        )

        df[col + "_day"] = df[col].apply(lambda x: x.day)
        # handle missing values in year column
        df[col + "_day"] = df[col + "_day"].apply(
            lambda x: int(df[col + "_day"].mean()) if pd.isna(x) else int(x)
        )
        # last, scale the new column so that it is ready for models
        df[col + "_day"] = (
            df[col + "_day"] - df[col + "_day"].min()
        ) / (
            df[col + "_day"].max() - df[col + "_day"].min()
        )

    df.drop(columns=DATETIME_COLUMNS, inplace=True)


def percentage_cols_transform(x):
    """
    Desc.
    """
    if type(x) == float:
        return x
    else:
        x = x.replace("%", "")
        return float(x) / 100


def handle_percentage_features(df: pd.DataFrame):
    """
    This column is special. It is numerical but involved percentages. We need to format it apart.
    """
    for col in PERCENTAGE_COLUMNS:
        df[col] = df[col].apply(percentage_cols_transform)
    return df


def drop_univalued_columns(df: pd.DataFrame):
    """
    Desc.
    """
    df.drop(columns=UNIVALUED_COLUMNS, inplace=True)
    df.drop(columns=["Square Feet"], inplace=True) # this columns is multivalued but has got too many missing data


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


def fill_na_mean(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Desc.
    """
    for col in NUMERICAL_FEATURES:
        try:
            mean = X_train[col].mean()
        except TypeError: # column is encoded using strings
            mean = np.mean(X_train[col].apply(lambda x: float(x)))
        X_train[col] = X_train[col].apply(lambda x: mean if(pd.isna(x)) else x)
        X_test[col] = X_test[col].apply(lambda x: mean if(pd.isna(x)) else x)
    return X_train, X_test


def fill_na_median(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Desc.
    """
    for col in NUMERICAL_FEATURES:
        try:
            med = X_train[col].median()
        except TypeError: # column is encoded using strings
            med = np.median(X_train[col].apply(lambda x: float(x)))
        X_train[col].apply(lambda x: med if(pd.isna(x)) else x)
        X_test[col].apply(lambda x: med if(pd.isna(x)) else x)
    return X_train, X_test


def fill_categorical_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Desc.
    """
    for col in CATEGORICAL_COLUMNS:
        # apply custom treatments depending on columns
        # TO DO
        most_frequent_value = X_train[col].mode()[0]
        X_train[col] = X_train[col].apply(lambda x: most_frequent_value if pd.isna(x) else x)
        X_test[col] = X_test[col].apply(lambda x: most_frequent_value if pd.isna(x) else x)
    """
    for col in HYBRID_FEATURES:
        most_frequent_value = X[col].mode()[0]
        X[col] = X[col].apply(lambda x: most_frequent_value if pd.isna(x) else x)
    """

    return X_train, X_test


def scale_numerical_features(X_train: pd.DataFrame, X_test: pd.DataFrame, method="minmax"):
    """
    Desc.
    """
    if method not in ["standard", "minmax"]:
        raise UnknownScalingMethodError
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()

    # scale the features
    X_train[NUMERICAL_FEATURES] = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
    X_test[NUMERICAL_FEATURES] = scaler.transform(X_test[NUMERICAL_FEATURES])
    return X_train, X_test

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


def apply_label_encoding(X: pd.DataFrame, full=False, use_sklearn=False):
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
           mapping = {
                "a few days or more": 0,
                "within a day": 1,
                "within a few hours" : 2,
                "within an hour": 3
           }
        elif col == "Room Type":
            mapping = {
                "Shared room": 0,
                "Private room": 1,
                "Entire home/apt": 2
            }
        elif col == "Room Type":
            mapping = {
                "Shared room": 0,
                "Private room": 1,
                "Entire home/apt": 2
            }
        else:
            mapping = None

        if use_sklearn or (mapping is None): # beware! unable to specify the order of categories using OrdinalEncoder yet
            encoder = OrdinalEncoder()
            encoded_col = encoder.fit_transform(X[[col]])
            X[col] = encoded_col.reshape(-1, 1)
        else: # encode only the ordinal features using a specified mapping
            X[col] = X[col].apply(lambda x: mapping[x] if x in mapping.keys() else np.nan)

    return X
    

def encode_boolean_features(X: pd.DataFrame):
    """
    Encode boolean features using a binary map 0:1.
    Note: binary maps can be viewed as a two-class ordinal encoding.
    """
    
    for col in BOOLEAN_FEATURES:
        if col == "Is Superhost": # specify the order to binary map
           mapping = {
                "f": 0,
                "t": 1
           }
        elif col == "Is Exact Location":
            mapping = {
                "f": 0,
                "t": 1
            }
        elif col == "Instant Bookable":
            mapping = {
                "f": 0,
                "t": 1
            }
        elif col == "Business Travel Ready":
            mapping = {
                "f": 0,
                "t": 1
            }
        else:
            mapping = None

        if mapping is None:
            raise UnspecifiedMappingError
        else:
            X[col] = X[col].apply(lambda x: mapping[x] if x in mapping.keys() else np.nan)

    return X


def ensure_numerical_columns_type(X: pd.DataFrame):
    """
    Make sure all numerical columns are of type float.
    """
    X[NUMERICAL_FEATURES] = X[NUMERICAL_FEATURES].applymap(float)
    return X


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Desc.
    """
    y = np.ravel(y)
    if X.shape[0] != y.shape[0]:
        raise UnalignedDataFramesError
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=X[""], shuffle=True)


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

    # drop the columns that take only a single value
    drop_univalued_columns(df)

    # then, convert columns involving dates to the proper format
    df = process_datetime_columns(df)

    # and split each date-based column into 3 components (day, month, year) + handle NaNs
    datetime_to3columns(df)

    # apply a special treatment to the host_response_rate column, which involves percentages
    df = handle_percentage_features(df)

    # extract the identifiers from the data (columns needed to identify a given sample, like ID, Name...)
    (data, df_identifiers) = extract_identifiers(df)

    # extract the target variable from the data
    X, y = extract_target(data)

    # now, we split the features and target into a training and a test set
    (X_train, X_test, y_train, y_test) = split_data(X, y)

    # at this step, we fill the missing values in the features. 2 strategies are allowed
    if fill_na_method not in ["mean", "median"]:
        raise UnknownFillMethodError
    elif fill_na_method == "mean":
        X_train, X_test = fill_na_mean(X_train, X_test)
    elif fill_na_method == "median":
        X_train, X_test = fill_na_median(X_train, X_test)

    # replace NaN values for the categorical columns too, using the majority value
    X_train, X_test = fill_categorical_features(X_train, X_test)

    # scale numerical features
    X_train, X_test = scale_numerical_features(X_train, X_test)
    
    # now, we apply encoding to the categorical features
    if encoder_method not in ["classical", "FullLabelEncoder", "FullOneHotEncoder"]:
        raise UnknownEncodingMethodError
    elif encoder_method == "classical":
        X_train = apply_one_hot_encoding(X_train, full=False)
        X_train = apply_label_encoding(X_train, full=False)
        X_test = apply_one_hot_encoding(X_test, full=False)
        X_test = apply_label_encoding(X_test, full=False)
    # WARNING: do not use the following two if you do not know what you are doing, use the default value instead
    elif encoder_method == "FullOneHotEncoder": # WARNING: be very careful when we use the following
        X_train = apply_one_hot_encoding(X_train, full=True)
        X_test = apply_one_hot_encoding(X_test, full=True)
    elif encoder_method == "FullLabelEncoder": # WARNING: be very careful when we use the following
        X_train = apply_label_encoding(X_train, full=True, use_sklearn=True)
        X_test = apply_label_encoding(X_test, full=True, use_sklearn=True)
    
    # last, encode the boolean features using a binary map
    X_train = encode_boolean_features(X_train)
    X_test = encode_boolean_features(X_test)

    # ensure all numerical columns are of type float
    X_train = ensure_numerical_columns_type(X_train)
    X_test = ensure_numerical_columns_type(X_test)

    return (X_train, y_train, X_test, y_test, df_identifiers)

def run_preprocessing(store_copy=False):
    """
    Automation for the preprocessing part.
    """
    # load the data
    df = load_data()
    # apply the preprocessing
    (X_train, y_train, X_test, y_test, df_identifiers) = preprocessing(df)
    if store_copy:
        # dump it into a csv for further investigation
        X_train.to_csv('X_train_processed.csv', header=True, columns=X_train.columns)
        X_test.to_csv('X_test_processed.csv', header=True, columns=X_test.columns)
    
    return (X_train, y_train, X_test, y_test, df_identifiers)


if __name__ == "__main__":
    (X_train, y_train, X_test, y_test, df_identifiers) = run_preprocessing(store_copy=True)