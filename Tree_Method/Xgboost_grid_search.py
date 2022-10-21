#############################################################################
#############################################################################
#############################################################################
"""
                                ML Projet
3A Centrale Supelec
Date : 18/10/2022
Author : Tristan BASLER
Description :
    Using XGBOOST model to predict price for an AirBnB appartement
                                                                          """
#############################################################################
#############################################################################
#############################################################################

# Importation
import sys
sys.path.insert(0, "../") # To insert the package on the path
import numpy as np
import os
import datetime
import preprocessing.preprocessing as preprocess
from sklearn.metrics import mean_squared_error
import xgboost as xg
from Tree_Method.utils import writing_description
from sklearn.model_selection import GridSearchCV

# Params definition
FILL_NA_METHOD = None
ENCODER_METHOD = "LabelEncoder"
DATEITME_TREATMENT = 'Linearization'

# Save Folder initialization
save_file_path = "../logs/xgboost_grid_search/{}/".format(
    datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(save_file_path, exist_ok=True)

# Preprocessing
data = preprocess.load_data()
(X_train, y_train, X_test, y_test, df_identifiers) = preprocess.preprocessing(data,
                                                                              encoder_method = ENCODER_METHOD,
                                                                              fill_na_method = FILL_NA_METHOD,
                                                                              datetime_treatment = DATEITME_TREATMENT)

# Learning

estimator = xg.XGBRegressor(objective="reg:squarederror", nthread=4, seed=42)

parameters = {
    "max_depth": [3, 4],
    "learning_rate": [0.1, 0.05],
}


grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring="neg_root_mean_squared_error",
    n_jobs=10,
    cv=10,
    verbose=True,
)

grid_search.fit(X_train, y_train)

grid_search.best_estimator_
grid_search.best_params_
grid_search.best_score_

print(
    "Root Mean Squared Error is: ",
    np.sqrt(mean_squared_error(y_test, grid_search.best_estimator_.predict(X_test))),
)

grid_search.best_estimator_.save_model(save_file_path + "/model.json")

description = {}
description["Date"] = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
description["User"] = os.getcwd().split("\\")[2]
description["Model"] = "Xgboost Grid Search"
description["Metrics"] = "Root Mean Squared Error is: {}".format(
    np.sqrt(mean_squared_error(y_test, grid_search.best_estimator_.predict(X_test)))
)
description[" \n Params Description"] = ""
description.update(grid_search.best_params_)
writing_description(description, save_file_path + "config.txt")
