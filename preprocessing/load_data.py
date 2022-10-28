"""
This script loads the data from the cloned git repository.
"""

import os
import git

# Pandas is used to manipulate tabular data.
import pandas as pd


def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


def load_data():
    """
    Load the data from the data/ folder.
    """
    # define path constants
    DATA_FOLDER = "data"
    PATH_CSV = "train_airbnb_berlin.csv"

    # read the training data
    df = pd.read_csv(
        os.path.join(
            get_git_root(os.getcwd()),
            DATA_FOLDER,
            PATH_CSV
        )
    )
    return df

