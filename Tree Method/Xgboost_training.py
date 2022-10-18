# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:35:12 2022

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


## Learning

model = xg.XGBRegressor(
    learning_rate=0.001,
    n_estimators=6000,
    max_depth=4,
    min_child_weight=0,
    gamma=0.6,
    subsample=0.7,
    colsample_bytree=0.7,
    objective="reg:squarederror",
    nthread=-1,
    scale_pos_weight=1,
    seed=27,
    reg_alpha=0.00006,
    random_state=42,
)

model.fit(X_train, y_train)

yhat = model.predict(X_test)


print("Root Mean Squared Error is: ", np.sqrt(mean_squared_error(y_test, yhat)))

# model_xgb_2 = xgb.Booster()
# model_xgb_2.load_model("model.json")

## Saving and Metrics

save_file_path = "../logs/xgboost/{}/".format(
    datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(save_file_path, exist_ok=True)
model.save_model(save_file_path + "/model.json")


description = {}
description["Date"] = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
description["User"] = os.getcwd().split("\\")[2]
description["Model"] = "Xgboost"
description["Metrics"] = "Root Mean Squared Error is: {}".format(
    np.sqrt(mean_squared_error(y_test, yhat))
)

description[" \n Params Description"] = ""
description.update(model.get_params())
writing_description(description, save_file_path + "config.txt")
