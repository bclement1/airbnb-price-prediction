"""
Preprocessing.py
"""

# Python ≥3.5 is required
import sys
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
                    TRAINING_CSV
                    )
                )
    print(training_data.head())
    return training_data, test_data

if __name__ == "__main__":
    training_data, test_data = load_data()