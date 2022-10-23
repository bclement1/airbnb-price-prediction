"""
Perform Linear Regression on the data and print the RMSE.
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import SGDRegressor # linear regression using stochastic gradient descent
from sklearn.metrics import mean_squared_error
import sys
sys.path.insert(0, '../../preprocessing') # fetch the preprocessing module
from preprocessing import run_preprocessing

def run_linear_regression(learning_rate=0.01, alpha=1, penalty='l2', verbose=False):
    """
    Fit a linear regression model (with intercept) on training data and apply it on the test data.
    Print RMSE.
    """
    # get the preprocessed data
    (X_train, y_train, X_test, y_test, df_identifiers) = run_preprocessing()

    # create the model
    lin_reg = SGDRegressor(
        loss='squared_error',
        penalty=penalty, # by default, we are performing a Ridge regression
        eta0=learning_rate, # learning rate
        learning_rate='constant',
        max_iter=1000,
        alpha=alpha # constant multiplying the regularization part
    )

    # fit the model on the training data
    lin_reg.fit(X_train, y_train)

    # just to have a clue, get the predictions 
    y_train_pred = lin_reg.predict(X_train)
    RMSE_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # perform actual prediction on the test set
    y_pred = lin_reg.predict(X_test)
    RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))

    if verbose: # print info on the fitted model
        print("Mean of the target variable:")
        print(y_train.mean())
        print("\nRMSE on training set:")
        print(RMSE_train)
        print("\nRMSE on test set:")
        print(RMSE_test)
        """
        print("\n*** LINEAR REGRESSION USING SGD AND L2-PEN")
        print("\n*** Coefficient: ***\n")
        print(lin_reg.coef_)
        print("\n*** Intercept: ***\n")
        print(lin_reg.intercept_)
        print("\n*** Features used in the model: ***\n")
        print(lin_reg.feature_names_in_)
        """
        """
        important note: why is feature_names_not showing all features?
        that is because the linear regressor trained using a L2 penalization!
        therefore, a trade-off between the model's performance and the model's
        simplicity was reached during the training phase, leading to set aside
        some of the features (the least informative ones)
        """
    return (RMSE_train, RMSE_test)

if __name__ == "__main__":
    run_linear_regression(verbose=True)


