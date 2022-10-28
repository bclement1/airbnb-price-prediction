"""
Perform Linear Regression on the data and print the RMSE.
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error
from svr import run_svm

def run_grid_search_svr(kernel="linear", verbose=True):
    """
    Fit a linear regression model (with intercept) on training data and apply it on the test data.
    Print RMSE.
    """

    # define the range of parameter C
    Cs = [5, 10, 20, 50, 100]
    coef0s = [1, 2, 3, 4, 5]

    SCORES = {}
    best_C=None
    best_coef0=None
    best_RMSE_test=np.inf

    # start main loop for linear kernel
    print("*** LINE SEARCH FOR SVM: ***")
    if kernel=="linear":
        for C in Cs:
            print("C =", C)
            RMSE_test = run_svm(C=C, kernel="linear", verbose=verbose)
            print("RMSE on test:", RMSE_test, "\n")
            if RMSE_test < best_RMSE_test:
                best_RMSE_test = RMSE_test
                best_C = C
                SCORES[C] = RMSE_test
        print("The best score was found for C={}".format(best_C))
    elif kernel=="poly":
        for C in Cs:
            for coef0 in coef0s:
                print("C =", C)
                print("coef0 =", coef0)
                RMSE_test = run_svm(C=C, kernel="poly", coef0=coef0, verbose=verbose)
                print("RMSE on test:", RMSE_test, "\n")
                if RMSE_test < best_RMSE_test:
                    best_RMSE_test = RMSE_test
                    best_C = C
                    best_coef0 = coef0
        print("The best score was found for C={}, coef0={}".format(best_C, best_coef0))
    
    print("\nRMSE on test set:", best_RMSE_test)

if __name__ == "__main__":
    run_grid_search_svr(kernel="poly", verbose=True)


