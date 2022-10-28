"""
Perform Linear Regression on the data and print the RMSE.
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVR # we will use the SVM implementation for regression from Scikit-Learn
from sklearn.metrics import mean_squared_error
import sys
sys.path.insert(0, '../../preprocessing') # fetch the preprocessing module
from preprocessing import run_preprocessing

def run_svm(kernel="linear", C=10**3, coef0=None, verbose=False):
    """
    Fit a linear regression model (with intercept) on training data and apply it on the test data.
    Print RMSE.
    """
    # get the preprocessed data
    (X_train, y_train, X_test, y_test, df_identifiers) = run_preprocessing()

    # instanciate a Support Vector Machine model for regression using the hyperparameter passed in argument
    if kernel=="linear":
        svm = SVR(kernel=kernel, C=C)
    elif kernel=="poly":
        svm = SVR(kernel=kernel, C=C, coef0=coef0)
    # fit the classifier
    svm.fit(X_train, y_train)

    # just to have a clue, get the predictions 
    #y_train_pred = svm.predict(X_train)
    #RMSE_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # perform actual prediction on the test set
    y_pred = svm.predict(X_test)
    RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))

    if verbose: # print info on the fitted model
        print("Mean of the target variable:")
        print(y_train.mean())
        #print("\nRMSE on training set:")
        #print(RMSE_train)
        print("\nRMSE on test set:")
        print(RMSE_test)
    
    return RMSE_test

if __name__ == "__main__":
    run_svm(verbose=True)


