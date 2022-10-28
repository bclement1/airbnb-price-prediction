"""
Perform Linear Regression on the data and print the RMSE.
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error
from linear_regression import run_linear_regression

def run_grid_search_linear_regression(penalty='l2', verbose=True):
    """
    Fit a linear regression model (with intercept) on training data and apply it on the test data.
    Print RMSE.
    """

    # define the parameters of the grid
    learning_rates = [1e-3] #np.logspace(-5, -1, 5)
    alphas = [1e-6] #np.logspace(-7, -3, 5)

    SCORES = {}
    best_learning_rate=None
    best_alpha=None
    best_RMSE_test=np.inf

    # start main loop
    print("*** STARTING GRID SEARCH: ***")
    for learning_rate in learning_rates:
        print("\nlearning_rate=", learning_rate)
        for alpha in alphas:
            print("alpha=", alpha)
            (_, RMSE_test) = run_linear_regression(
                learning_rate=learning_rate,
                alpha=alpha,
                penalty=penalty,
                verbose=True
            )
            print("RMSE on test:", RMSE_test, "\n")
            if RMSE_test < best_RMSE_test:
                best_RMSE_test = RMSE_test
                best_learning_rate = learning_rate
                best_alpha = alpha
            SCORES[(learning_rate, alpha)] = RMSE_test
    
    print("The best score was found for alpha={} and learning_rate={}".format(
                                                                        best_alpha,
                                                                        best_learning_rate
                                                                    ))

    print("\nRMSE on test set:")
    print(best_RMSE_test)


if __name__ == "__main__":
    run_grid_search_linear_regression(verbose=True)


