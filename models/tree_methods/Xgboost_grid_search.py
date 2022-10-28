# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:36:13 2022

@author: trisr
"""
import sys

sys.path.insert(0, "../")

import numpy as np
import os
import datetime
import preprocessing.preprocessing as preprocess
from sklearn.metrics import mean_squared_error
import xgboost as xg
from sklearn.model_selection import GridSearchCV


def writing_description(description, filename):
    with open(filename, "w") as f:
        for key, value in description.items():
            f.write("%s : %s \n" % (key, value))


## xgboost

## Specific Preprocessing
data = preprocess.load_data()
(X_train, y_train, X_test, y_test, df_identifiers) = preprocess.preprocessing(data)

datetime_columns = ["Host Since", "First Review", "Last Review"]
for col in datetime_columns:
    X_train[col + "_year"] = X_train[col].apply(lambda x: x.year)
    X_train[col + "_month"] = X_train[col].apply(lambda x: x.month)
    X_train[col + "_day"] = X_train[col].apply(lambda x: x.day)
X_train.drop(columns=datetime_columns, inplace=True)

datetime_columns = ["Host Since", "First Review", "Last Review"]
for col in datetime_columns:
    X_test[col + "_year"] = X_test[col].apply(lambda x: x.year)
    X_test[col + "_month"] = X_test[col].apply(lambda x: x.month)
    X_test[col + "_day"] = X_test[col].apply(lambda x: x.day)
X_test.drop(columns=datetime_columns, inplace=True)

na_index = y_train[y_train.isna()].index
X_train.drop(na_index, inplace=True, errors="ignore")
y_train.drop(na_index, inplace=True, errors="ignore")


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


save_file_path = "../logs/xgboost_grid_search/{}/".format(
    datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(save_file_path, exist_ok=True)
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
